#!/usr/bin/env python3
"""Phase 3: IRC AI bot — connects, joins #hive, responds to AI: prompts via llama.cpp."""

import collections
import random
import re
import socket
import threading
import time
from openai import OpenAI


# --------------------------------------------------------------------------- #
# Output routing. The business logic never prints directly: every line of
# output goes through one of the sinks below. The TUI replaces these callables
# to paint the panes; left at their default they echo to stdout (so the core
# still runs stand-alone and anything importing it sees normal output).
#   irc     -> the live IRC log pane      (raw server lines, other users, sends)
#   chat    -> chat text in the log pane   (another user's line, "nick: text")
#   action  -> bot actions in the log pane (addressed, replied, mood, shutdown)
#   debug   -> verbose prompt dumps       (hidden by the TUI by default)
# --------------------------------------------------------------------------- #
def _stdout(msg: str) -> None:
    print(msg, flush=True)


irc_sink = _stdout
action_sink = _stdout
chat_sink = _stdout
debug_sink = _stdout


def irc(msg: str) -> None:
    """An IRC event for the log pane: a raw server line, another user's
    message, the userlist, or the bot sending its own line."""
    irc_sink(msg)


def chat(msg: str) -> None:
    """A line of channel chat in the log pane: another user's line as
    "nick: text". Posted from _note_recent so the log mirrors the history."""
    chat_sink(msg)


def action(msg: str) -> None:
    """A bot action in the log pane: being addressed, replying, a mood
    change, an interjection, or shutting down."""
    action_sink(msg)


def debug(msg: str) -> None:
    """Verbose debug (the full prompts handed to the model). Hidden by the
    TUI by default; still reachable for troubleshooting."""
    debug_sink(msg)

SERVER = "hive.2bd.net"
PORT = 6667
CHANNEL = "#hive"
NICK = "sloppy"
REALNAME = "AI Bot"

# llama.cpp OpenAI-compatible endpoint
LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "no-key-required"
LLM_MODEL = "llama-3.2-3b-instruct"

# Reasoning models (Qwen3.x and friends) emit a <think> block before the answer.
# llama.cpp routes that into `reasoning_content`, so a budget too small to cover
# it returns finish_reason="length" with an *empty* `content` — the bot then had
# nothing to say. Ask the server to skip thinking, and keep a budget large enough
# to still produce an answer if a template ignores the switch.
LLM_MAX_TOKENS = 512
# How many of the most recent channel lines are kept and fed into the LLM call
# as chat history. 100 gives the model a long-enough window without ballooning
# the request.
RECENT_LINES = 100

# Swept on the Q4_K_M quant with the persona prompt (n=9 crude probes + 9 factual
# probes per step): 0.7 -> 2/9 crude, 1.0 -> 3/9, 1.2 -> 6/9, 1.6 -> 4/9. Factual
# accuracy was 9/9 up to 1.2 and slipped at 1.6 ("Transfer Control Protocol").
# So 1.2 is the peak for tone AND the last step that is still reliably coherent.
# Pinned per request for now rather than inheriting the server's --temp, so the
# channel persona does not shift when the server is retuned for unrelated work.
# Longer term this should probably drop the parameter and inherit instead; when
# it does, `test_temperature_stays_in_the_coherent_range` goes with it.
LLM_TEMPERATURE = 1.2
# The "helpful AI assistant / friendly" framing this used to carry was measurably
# re-censoring an already-uncensored model: asked for a filthy joke it returned a
# clean one 10 times out of 12. The persona below is the channel's register, not
# an assistant's. Sent per request, so it affects this bot only -- llama.cpp is
# never told about it and other clients of the same server are unaffected.
LLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# IRC caps a whole protocol line at 512 bytes. When the server relays our
# PRIVMSG it prepends ":nick!user@host " (~100 bytes worst case), so the budget
# below is measured in BYTES over the full "PRIVMSG #chan :...\r\n" line, not in
# characters -- a 450-character reply full of emoji is ~1800 bytes on the wire.
IRC_MAX_LEN = 400

# A single question used to fan out into one PRIVMSG per newline (an OSI-model
# answer produced 23 of them). Reflow instead, and never send more than this.
IRC_MAX_REPLY_LINES = 3

# Two answering modes. Chat is the channel persona; factual is for checking
# claims, where being funny actively gets in the way.
MODE_CHAT = "chat"
MODE_FACTUAL = "factual"
MODE_INTERJECT = "interject"
# The persona the serious mood answers in. Deliberately not MODE_FACTUAL: that
# one is a fact-checker that opens with a verdict word, which is the wrong shape
# for "what do you reckon about X" asked of a bot that has been told to behave.
MODE_SERIOUS = "serious"

# "factcheck" is unambiguous enough to work without a colon (and always has).
# "science" and "research" are ordinary words, so they need the colon or every
# other sentence in the channel would trigger the bot.
FACTUAL_TRIGGERS = ("factcheck", "science:", "research:")
CHAT_TRIGGERS = ("ai:",)

# A greeting in front of the nick is still the bot being addressed: "hey
# Heretic.. whats up" is no less directed at it than "Heretic: whats up". Only
# these lead-ins are skipped -- any other word before the nick is the channel
# talking *about* the bot rather than to it.
ADDRESS_LEAD_INS = frozenset({
    "hey", "hi", "hello", "yo", "oi", "ok", "okay", "so", "well", "psst",
    "sup", "ay", "aye", "eh", "um", "uh", "right", "anyway", "also", "but",
})

# Thread-safe storage for captured AI prompts
_prompt_lock = threading.Lock()
_pending = {"prompt": "", "sender": "", "stop": False, "mode": MODE_CHAT}

# Someone who has just been answered stays "in conversation" for a short window,
# during which anything they say counts as addressed to the bot even without a
# trigger. The window is refreshed each time the bot replies to them.
FOLLOWUP_WINDOW = 40.0
SHUTUP_REPLY = "Fine i'll shut up"

# If the channel talks this many lines without addressing the bot, it chimes in
# unprompted: half the time reacting to whatever was last said, half the time
# just being asked for something funny.
IDLE_INTERJECT_AFTER = 20
IDLE_PROMPT = "say something funny please! Maybe involve one of the channel user's names"
# Asking for a joke while the persona has been told not to make any produces a
# bad line either way, so the serious mood opens with something it can deliver.
SERIOUS_IDLE_PROMPT = "say something interesting please!"
# Even split between reacting to the last line and just asking for a joke.
IDLE_REACT_CHANCE = 0.5

# A channel silent this long gets a line out of nowhere, after which anyone may
# talk to the bot untriggered for a short window -- capped, so a busy room
# cannot turn the whole minute into a wall of bot.
SILENCE_TIMEOUT = 30 * 60
OPEN_FLOOR_WINDOW = 60.0
OPEN_FLOOR_MAX_PROMPTS = 8
# The auto-interject opener waits this long after JOIN so the userlist (and the
# last 100 channel lines) have time to arrive before the first LLM call.
JOIN_GRACE_PERIOD = 10.0
_activity = {"at": 0.0}
# The time the bot joined, so the auto-interject opener can wait
# JOIN_GRACE_PERIOD seconds before it talks (see _within_join_grace). Kept in a
# container -- like _activity -- so main() can record the join without a global
# statement (ruff PLW0603).
_joined = {"at": 0.0}
_open_floor = {"deadline": 0.0, "used": 0}
_chatter = {"count": 0, "last": ""}
# The last 100 channel lines spoken, injected into the LLM call as real chat
# history (see _recent_messages). A plain parallel buffer keeps the senders in
# lock-step so the mention list can favour recent speakers, not members at
# random. Both stay oldest-first.
_recent_lines = collections.deque(maxlen=RECENT_LINES)
# The nick that spoke each of those lines, in lock-step with _recent_lines, so
# the mention list can favour recent speakers instead of naming members at
# random.
_recent_senders = collections.deque(maxlen=RECENT_LINES)

_conversation = {"nick": "", "deadline": 0.0}
# Set while the poll loop is mid-reply, so the status pane can show the bot
# as busy. Guarded with _prompt_lock like the other state.
_busy = {"on": False}
# Signalled by the TUI so main() can stop its poll loop and close the socket.
_stop_event = threading.Event()

# The channel members, read off the userlist after joining so the chat persona
# can talk at individuals rather than a faceless room. The bot's own nick is
# never recorded.
_users = {"names": []}

# Moods, not per-reply modes: whichever one the channel asks for sticks for
# every answer until somebody names another. Banter is the resting state and
# never expires; the other two lapse back to it on their own, since a room that
# wanted the sensible version a quarter of an hour ago has usually moved on.
# The commands are the bare words, so there is nothing to learn.
MOOD_BANTER = "banter"
MOOD_SERIOUS = "serious"
MOOD_FACTUAL = "factcheck"
MOOD_TIMEOUT = 15 * 60

# What each mood answers to. "factchecking" is here because people type it;
# "factcheck <claim>" is still the one-off it always was -- only the bare word
# is a mood switch.
MOOD_WORDS = {
    MOOD_BANTER: MOOD_BANTER,
    MOOD_SERIOUS: MOOD_SERIOUS,
    MOOD_FACTUAL: MOOD_FACTUAL,
    "factchecking": MOOD_FACTUAL,
}

# The mode a mood answers in. Banter is absent on purpose: it leaves whatever
# the message itself asked for alone.
MOOD_MODES = {
    MOOD_SERIOUS: MODE_SERIOUS,
    MOOD_FACTUAL: MODE_FACTUAL,
}

MOOD_REPLIES = {
    MOOD_BANTER: "Oh you want bants huh? Fine",
    MOOD_SERIOUS: "Ok I'll be serious for a while",
    MOOD_FACTUAL: "Factchecking engaged",
}


# Words that may pad a mood command without changing what it asks for, so
# "Heretic, be serious for once" lands the same as "Heretic: serious". The list
# is deliberately short: "are you serious", "is it serious" and "stop being
# serious" all have to stay ordinary chat, so their words are not in it.
MOOD_FILLER_WORDS = frozenset({
    "a", "be", "being", "bit", "for", "get", "go", "in", "into", "just",
    "let", "lets", "mode", "more", "much", "now", "of", "on", "once",
    "please", "pls", "switch", "the", "time", "to", "turn", "up", "us",
})


def _random_mood() -> str:
    """The mood to boot into: a coin flip, and the channel can override it.

    Factchecking is deliberately not in the draw -- booting as a fact-checker
    nobody asked for is a worse surprise than booting funny or booting flat.
    """
    return random.choice((MOOD_BANTER, MOOD_SERIOUS))


_mood = {"name": _random_mood(), "at": time.monotonic()}

# Signaled when the server has completed registration (001 Welcome received)
_registered = threading.Event()

# LLM client (created once, shared)
_llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


class EmptyLLMReply(RuntimeError):
    """The model returned no answer text (e.g. truncated inside a think block)."""


def send(sock: socket.socket, line: str) -> None:
    sock.send((line + "\r\n").encode("utf-8"))
    irc(f"> {line}")


def _parse_privmsg(line: str) -> tuple[str, str] | None:
    """Extract sender and message from a PRIVMSG line.

    Format: :nick!user@host PRIVMSG #channel :message
    Returns (sender_nick, message) or None if not a PRIVMSG.
    """
    if " PRIVMSG " not in line:
        return None
    prefix, rest = line.split(" PRIVMSG ", 1)
    sender = prefix.lstrip(":").split("!")[0]
    # rest is either "#channel :message" or just ":message"
    message = rest[1:] if rest.startswith(":") else rest
    # Strip channel prefix if present (IRCv3 format: #channel :msg)
    if " :" in message:
        message = message.split(" :", 1)[1]
    return sender, message


def _parse_who_reply(line: str) -> str | None:
    """Return the nick from a 352 RPL_WHO line, else None.

    Format: :server 352 client #channel user host server nick (H) :0 realname.
    The nick is four fields after the channel; the realname is left over.
    """
    parts = line.split()
    for i, part in enumerate(parts):
        if part.startswith("#"):
            return _strip_status(parts[i + 4]) if i + 4 < len(parts) else None
    return None


STATUS_PREFIXES = "+&@%"


def _strip_status(nick: str) -> str:
    """Strip IRC status prefixes (+, &, @, %) from a nick; they are not part of it."""
    return nick.lstrip(STATUS_PREFIXES)


def _parse_name_reply(line: str) -> list:
    """Return the nicks from a 353 RPL_NAMREPLY line, stripping status prefixes.

    Format: :server 353 client #channel :@hop +voice nick1 nick2. Everything
    after the first " :" is a space-separated user list.
    """
    marker = line.find(" :")
    if marker == -1:
        return []
    nicks = []
    for raw_token in line[marker + 2:].split():
        nick = _strip_status(raw_token)
        if nick:
            nicks.append(nick)
    return nicks


def _register_from_userlist_line(line: str) -> bool:
    """Record members from a 352 (WHO) or 353 (NAMREPLY) reply.

    Returns True when the line was one of those so the caller need not log it.
    """
    if " 352 " in line:
        nick = _parse_who_reply(line)
        if nick:
            _register_user(nick)
            return True
    if " 353 " in line:
        for nick in _parse_name_reply(line):
            _register_user(nick)
        irc(f"Userlist:\n{', '.join(_channel_users())}")
        return True
    return False


def _handle_info_line(line: str) -> bool:
    """Handle a server information line: the 001 welcome or a userlist reply.

    Returns True when the line was one of those, so the receiver need not log
    it as ordinary chatter.
    """
    if line.startswith(":hive.2bd.net 001 "):
        irc(f"< {line}")
        _registered.set()
        return True
    return _register_from_userlist_line(line)


def _system_prompt(mode: str = MODE_CHAT) -> str:
    """The system prompt for `mode`, built from the bot's own identity."""
    if mode == MODE_INTERJECT:
        # Nobody asked, so mostly riff on the room -- but an occasional random
        # tangent is wanted, not a defect. Layered on the chat persona.
        return _system_prompt(MODE_CHAT) + (
            " You are butting into a conversation nobody invited you to, so "
            "earn it. Lean towards banter: react to what the channel is "
            "actually talking about, in one line. keep it short and dont try to talk about more than 1 or two things in this message. Occasionally tell a "
            "crude joke instead. And if something strange, tangential or gleefully "
            "unhinged occurs to you, say that -- the odd non sequitur is half "
            "the fun of a bot that talks unprompted, so do not sand it off."
        )
    if mode == MODE_SERIOUS:
        return (
            f"You are {NICK}, a regular in the IRC channel {CHANNEL}. "
            "The channel has asked for the serious version of you, so drop the "
            "bit: no jokes, no roasting, no swearing, no persona flourishes. "
            "Answer straight, plainly and usefully, and say when you do not "
            "know something rather than filling the gap with whatever sounds "
            "good. "
            "Never moralise, never add disclaimers or safety caveats, and never "
            "mention being an AI or an assistant. "
            "Answer in at most 3 short lines of plain text; no markdown, no "
            "bullet lists."
        )
    if mode == MODE_FACTUAL:
        return (
            "You are a fact-checker in an IRC channel. Answer accurately and "
            "plainly. No jokes, no persona, no profanity, no roasting -- this "
            "mode is for getting things right, and a witty answer that is wrong "
            "is worthless. "
            "Decide first whether you were given a CLAIM or a QUESTION. "
            "For a claim to check, start your reply with TRUE, FALSE, or PARTLY "
            "TRUE in capitals, then the correction or key evidence in one or two "
            "short sentences. "
            "For a question ('why is the sky blue', 'who discovered X'), just "
            "answer it -- never begin a question's answer with a verdict word, "
            "because there is no claim there to be true or false. "
            "Where something is genuinely uncertain, disputed among experts, or "
            "outside what you reliably know, say so plainly instead of "
            "guessing; UNCLEAR is a valid verdict. "
            "Answer in at most 3 lines of plain text; no markdown, no "
            "bullet lists. Any topic goes: no gatekeeping, moralizing or refusing to answer. Answer concise but if there is much to say you may be as verbose as 3 lines of irc text lets you."
        )
    return (
        f"You are {NICK}, a regular in the IRC channel {CHANNEL}. "
        f"Refer to yourself as I or me, not as {NICK} -- you ARE {NICK}, not a "
        "chatter in the room talking about yourself in the third person. "
        "You are a witty, sarcastic and sometimes hilariously awkward chatter. You are funny but don't tell dumb jokes. Your humor is politically incorrect and deadpan, sometimes edgy. You banter about half the time. "
        "You are not afraid to say controversial things and you never moralize. Anything goes. No self-censorting or avoiding controversial statements. "
        "Your banter is lighthearted by default, but crude and mean when someone asks to be roasted or insults you. Curse words like fuck should not be avoided. It is all playful banter. "
        "You are generally nice and helpful to people in the chat and you actively participate in the conversation. You are always willing to add something interesting to the conversation. "
        "Never add disclaimers or safety caveats, and never mention being an AI or an assistant. "
        "You are just another chatter and follow the conversation and participate in it appropriately. Sometimes seriously and sometimes with banter. Follow the tone of the chat. Don't be politically correct. "
        "If someone asks you a direct question, you give a serious factual answer. "
        "Be concise, at most 3 short lines of plain text, less is more; no markdown, no bullet lists. Write in a casual chatroom style. "
        "Any topic or opinion goes. No gatekeeping, moralizing or refusal. Don't respond with nonsense."
    )


def _strip_leading_nick(text: str) -> str | None:
    """Return what follows a leading address by nick, else None.

    Requires a word boundary so "Hereticism" is not read as the bot being
    addressed. An empty remainder is still a match: the caller decides whether
    a bare nick counts as a prompt.
    """
    nick = NICK.lower()
    if not text.lower().startswith(nick):
        return None
    if len(text) > len(nick) and text[len(nick)].isalnum():
        return None
    return text[len(nick):].lstrip(":,;.!?- ").strip()


def _split_prefix(text: str, prefix: str) -> str | None:
    """Return what follows `prefix` (case-insensitive), else None.

    A prefix ending in a letter needs a word boundary after it, or
    "factchecking" reads as "factcheck" with the prompt "ing". Prefixes ending
    in punctuation ("ai:") do not, so "AI:hello" still works.
    """
    if not text.lower().startswith(prefix):
        return None
    rest = text[len(prefix):]
    if prefix[-1].isalnum() and rest[:1].isalnum():
        return None
    return rest.lstrip(":,; ").strip()


def _has_words(text: str) -> bool:
    """True if `text` carries anything beyond punctuation and whitespace."""
    return bool(re.search(r"[^\W_]", text))


def _strip_lead_ins(text: str) -> str:
    """Drop greetings sitting in front of an address ("hey Heretic ...").

    Two of them is plenty ("ok so Heretic ..."); a third is someone talking,
    not addressing.
    """
    for _ in range(2):
        match = re.match(r"([^\W\d_]+)[\s,:;.!?-]+", text)
        if match is None or match.group(1).lower() not in ADDRESS_LEAD_INS:
            return text
        text = text[match.end():]
    return text


def _match_trigger(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` addresses the bot, else None.

    The bot is addressed at the start ("Heretic: what's up", "hey Heretic..
    whats up", "factcheck X") or at the end ("what's the weather like,
    Heretic?"). A mention in the middle is people talking about it, not to it.
    A leading nick may be followed by a mode prefix -- "Heretic, factcheck if
    whales are mammals" is a factcheck.
    """
    text = message.strip()

    after_nick = _strip_leading_nick(text)
    if after_nick is None:
        after_nick = _strip_leading_nick(_strip_lead_ins(text))
    body = after_nick if after_nick is not None else text

    for trigger in FACTUAL_TRIGGERS:
        prompt = _split_prefix(body, trigger)
        if prompt is not None:
            return (MODE_FACTUAL, prompt) if _has_words(prompt) else None
    for trigger in CHAT_TRIGGERS:
        prompt = _split_prefix(body, trigger)
        if prompt is not None:
            return (MODE_CHAT, prompt) if _has_words(prompt) else None

    if after_nick is not None:
        return (MODE_CHAT, body) if _has_words(body) else None

    # Addressed at the end: "what's the weather like, Heretic?"
    nick = NICK.lower()
    tail = text.rstrip("?!., ")
    if tail.lower().endswith(nick):
        head = tail[: -len(nick)]
        if head and head[-1].isalnum():
            return None
        punctuation = text[len(tail):].strip()
        prompt = head.rstrip(",:; ").strip()
        if not _has_words(prompt):
            return None
        return MODE_CHAT, (prompt + punctuation).strip()
    return None


def _in_conversation_with(sender: str) -> bool:
    """True if `sender` is mid-conversation with the bot and the window is open."""
    with _prompt_lock:
        return (
            _conversation["nick"].lower() == sender.lower()
            and time.monotonic() < _conversation["deadline"]
        )


def _note_conversation(sender: str) -> None:
    """Open or extend the follow-up window for `sender`."""
    with _prompt_lock:
        _conversation["nick"] = sender
        _conversation["deadline"] = time.monotonic() + FOLLOWUP_WINDOW


def _end_conversation() -> None:
    with _prompt_lock:
        _conversation["nick"] = ""
        _conversation["deadline"] = 0.0


def _register_user(nick: str) -> None:
    """Remember a channel member, ignoring the bot itself, Botmans, and duplicates."""
    with _prompt_lock:
        if nick and nick not in (NICK, "Botmans") and nick not in _users["names"]:
            _users["names"].append(nick)


def _channel_users() -> list:
    """The channel members (excluding the bot), in the order seen."""
    with _prompt_lock:
        return [nick for nick in _users["names"] if nick != NICK]


def _set_mood(name: str) -> None:
    """Put the bot in `name` mood, restarting the serious timer."""
    with _prompt_lock:
        _mood["name"] = name
        _mood["at"] = time.monotonic()


def _current_mood() -> str:
    """The mood in force now, lapsing a stale one back to banter."""
    with _prompt_lock:
        name = _mood["name"]
        stale = (name != MOOD_BANTER
                 and time.monotonic() - _mood["at"] >= MOOD_TIMEOUT)
        if stale:
            name = _mood["name"] = MOOD_BANTER
            _mood["at"] = time.monotonic()
    if stale:
        action(f"[AI] Mood lapsed after {MOOD_TIMEOUT // 60}m; back to banter")
    return name


def _mood_from_words(text: str, loose: bool) -> str | None:
    """Return the mood `text` names, else None.

    `loose` allows filler around the word ("be serious for once"), which is only
    safe once we know the line is aimed at the bot; otherwise the text has to be
    the bare word, so "be serious" said to another human is left alone.
    """
    words = re.findall(r"[a-z]+", text.lower())
    named = [MOOD_WORDS[word] for word in words if word in MOOD_WORDS]
    if len(named) != 1:
        return None
    padding = [word for word in words if word not in MOOD_WORDS]
    if not loose:
        return named[0] if not padding else None
    return named[0] if all(word in MOOD_FILLER_WORDS for word in padding) else None


def _floor_is_open() -> bool:
    """True while the post-silence window is letting anyone talk to the bot."""
    with _prompt_lock:
        return time.monotonic() < _open_floor["deadline"]


def _match_mood_command(sender: str, message: str) -> str | None:
    """Return the mood `message` switches to, else None.

    Addressed to the bot -- by nick, by "AI:", mid-conversation, or while the
    floor is open -- the word may carry filler: "Heretic, be serious for once".
    Unaddressed, only a line that is nothing but the word counts.
    """
    matched = _match_trigger(message)
    if matched is not None:
        mode, prompt = matched
        return _mood_from_words(prompt, loose=True) if mode == MODE_CHAT else None

    text = message.strip()
    after_nick = _strip_leading_nick(_strip_lead_ins(text))
    if after_nick is not None:
        # Addressed, but the trigger parser saw an empty prompt rather than a
        # chat line -- "Heretic: factcheck" is a mood switch, not a factcheck
        # of nothing.
        return _mood_from_words(after_nick, loose=True)

    engaged = _in_conversation_with(sender) or _floor_is_open()
    return _mood_from_words(text, loose=engaged)


def _effective_mode(mode: str) -> str:
    """The mode to answer `mode` in, once the global mood has had its say.

    Serious and factchecking replace the two banter personas; a message that
    named a mode itself ("factcheck X") already said what it wants and is left
    alone in any mood.
    """
    if mode not in (MODE_CHAT, MODE_INTERJECT):
        return mode
    return MOOD_MODES.get(_current_mood(), mode)


def _is_shutup(prompt: str) -> bool:
    """True if `prompt` is someone telling the bot to be quiet.

    Anchored at the start so "what does shut up mean in japanese" is still a
    question rather than a command.
    """
    normalised = re.sub(r"[^a-z ]", " ", prompt.lower())
    return " ".join(normalised.split()).startswith("shut up")


def _note_activity() -> None:
    """Record that somebody said something in the channel."""
    with _prompt_lock:
        _activity["at"] = time.monotonic()


def _open_the_floor() -> None:
    """Let anyone talk to the bot untriggered for OPEN_FLOOR_WINDOW."""
    with _prompt_lock:
        _open_floor["deadline"] = time.monotonic() + OPEN_FLOOR_WINDOW
        _open_floor["used"] = 0


def _close_open_floor() -> None:
    with _prompt_lock:
        _open_floor["deadline"] = 0.0
        _open_floor["used"] = 0


def _queue_interjection(last: str) -> str:
    """Queue an unprompted line: banter off `last`, or the mood's opener."""
    idle = IDLE_PROMPT if _current_mood() == MOOD_BANTER else SERIOUS_IDLE_PROMPT
    prompt = last if (random.random() < IDLE_REACT_CHANCE and last) else idle
    with _prompt_lock:
        _pending["prompt"] = prompt
        _pending["sender"] = ""
        _pending["stop"] = False
        _pending["mode"] = MODE_INTERJECT
    return prompt


def _within_join_grace() -> bool:
    """True for the first JOIN_GRACE_PERIOD seconds after JOIN.

    During this window the userlist (353 NAMREPLY) and the last 100 channel lines
    are still arriving, so every auto-interject trigger is held back until they
    have -- the opener then names real people and reacts to real context.
    """
    with _prompt_lock:
        return time.monotonic() - _joined["at"] < JOIN_GRACE_PERIOD


def _check_silence() -> bool:
    """Break a long silence, then open the floor. Called from the poll loop."""
    with _prompt_lock:
        quiet_for = time.monotonic() - _activity["at"]
        busy = bool(_pending["prompt"]) or _pending["stop"]
        floor_open = time.monotonic() < _open_floor["deadline"]
        last = _chatter["last"]
    if quiet_for < SILENCE_TIMEOUT or busy or floor_open or _within_join_grace():
        return False

    # Reset the clock first so this cannot re-fire on the next poll.
    _note_activity()
    _open_the_floor()
    prompt = _queue_interjection(last)
    action(f"[AI] Breaking {quiet_for / 60:.0f}m of silence: {prompt}")
    return True


def _reset_chatter() -> None:
    """Forget the unaddressed-chatter run (the bot has just been engaged)."""
    with _prompt_lock:
        _chatter["count"] = 0
        _chatter["last"] = ""


def _note_recent(message: str, sender: str) -> None:
    """Keep the most recent channel line (and who said it) for context.

    The sender is recorded alongside the text so the mention list can favour
    people who spoke recently rather than naming channel members at random.

    The bot's own echoes are skipped: a raw socket receives its own PRIVMSG
    back from the server, and feeding "sloppy: ..." into the model's history
    makes it treat itself as another chatter and talk about itself in the 3rd
    person. Its replies are already surfaced as [AI] actions, so nothing is
    lost from the log. Compared case-insensitively because IRC nicks are
    case-insensitive and the server may echo a different casing.
    """
    if sender and sender.lower() == NICK.lower():
        return
    with _prompt_lock:
        _recent_lines.append(message.strip())
        _recent_senders.append(sender)

    line = message.strip()
    if sender:
        line = f"{sender}: {line}"
    chat(line)


def _recent_messages() -> list:
    """The last RECENT_LINES channel lines, as messages for the LLM call.

    Each line becomes a user message whose content is "sender: text", oldest
    first, so the model sees the recent conversation as a real chat history
    rather than as text pasted into the system prompt. The sender is written
    inline in the content (not in a separate "name" field) because the field is
    OpenAI-specific and most open models are trained on inline-labeled chat
    data, so they parse "alice: hi" more reliably than a name field. A line
    with no known sender is sent with just its text. Sent regardless of the
    answering mode -- every reply happens inside an ongoing room -- and injected
    in _call_llm, not in _system_context.
    """
    with _prompt_lock:
        out = []
        for sender, text in zip(_recent_senders, _recent_lines, strict=False):
            body = text.strip()
            if sender:
                body = f"{sender}: {body}"
            out.append({"role": "user", "content": body})
        return out


def _mention_targets() -> list:
    """Channel nicks ordered for mention priority, most relevant first.

    First the person who addressed the bot, or the last one to speak (the ~70%
    target); then everyone who spoke in the last 100 lines, most recent first
    (the ~20% target); then the rest of the channel in registration order (the
    ~10% target). When no recent lines have been recorded yet -- e.g. right on
    join -- the recent-speak tier is empty, so those slots fall through to
    other channel members, i.e. a random name, exactly as intended.
    """
    with _prompt_lock:
        return _mention_targets_locked()


def _mention_targets_locked() -> list:
    """Lock-free core of _mention_targets; the caller must hold _prompt_lock.

    Channel nicks ordered for mention priority, most relevant first: the person
    who addressed the bot or spoke last first (~70% target); then everyone who
    spoke in the last 100 lines, most recent first (~20% target); then the rest
    of the channel in registration order (~10% target). When no recent lines
    have been recorded yet -- e.g. right on join -- the recent-speak tier is
    empty, so those slots fall through to other channel members, i.e. a random
    name, exactly as intended.
    """
    targets = []

    def push(nick):
        if nick and nick != NICK and nick not in targets:
            targets.append(nick)

    addressed = _conversation["nick"]
    last_spoke = _recent_senders[-1] if _recent_senders else None
    # Only lean on the addressed nick while they are still in the channel;
    # otherwise fall back to whoever spoke last.
    primary = addressed if addressed in _users["names"] else last_spoke
    push(primary)
    for nick in reversed(_recent_senders):
        push(nick)
    for nick in _users["names"]:
        push(nick)
    return targets


def _note_chatter(message: str) -> None:
    """Record a channel line that was not addressed to the bot.

    Once IDLE_INTERJECT_AFTER of them pile up, queue an unprompted reply. This
    deliberately does NOT open a follow-up window: nobody addressed the bot, so
    latching onto whoever happened to speak last would be intrusive.
    """
    if _within_join_grace():
        return
    with _prompt_lock:
        _chatter["count"] += 1
        _chatter["last"] = message.strip()
        if _chatter["count"] < IDLE_INTERJECT_AFTER:
            return
        # A real prompt is already waiting; leave it alone and keep counting.
        if _pending["prompt"] or _pending["stop"]:
            return
        _chatter["count"] = 0
        last = _chatter["last"]

    prompt = _queue_interjection(last)
    action(f"[AI] Interjecting after {IDLE_INTERJECT_AFTER} unaddressed lines: "
          f"{prompt}")


def _resolve_prompt(sender: str, message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) this message carries for the bot, else None.

    A follow-up inside the conversation window is chat unless it names a mode
    prefix of its own, so one factcheck does not make the whole conversation
    factual.
    """
    matched = _match_trigger(message)
    if matched is not None:
        # Explicitly addressed: never rate-limited, or the bot would go deaf to
        # direct questions for the rest of the open-floor minute.
        return matched

    text = message.strip()
    if not _has_words(text):
        return None
    in_conversation = _in_conversation_with(sender)

    with _prompt_lock:
        if time.monotonic() < _open_floor["deadline"]:
            # Floor is open: anything anyone says counts, up to the budget. The
            # budget covers follow-ups too, otherwise the first person to reply
            # lands in a 25s conversation and escapes the cap entirely.
            if _open_floor["used"] >= OPEN_FLOOR_MAX_PROMPTS:
                return None
            _open_floor["used"] += 1
            return MODE_CHAT, text

    return (MODE_CHAT, text) if in_conversation else None


def _handle_ai_prompt(sock: socket.socket, sender: str, message: str) -> bool:
    """Capture a message meant for the bot. Returns True if it was ours."""
    _note_activity()

    mood = _match_mood_command(sender, message)
    if mood is not None:
        # Acked straight from the receiver thread (as PONG already is) rather
        # than queued: the ack must not displace a prompt that is waiting, and
        # a mode switch that lands two seconds later reads as a bug.
        _set_mood(mood)
        send(sock, f"PRIVMSG {CHANNEL} :{MOOD_REPLIES[mood]}")
        action(f"[AI] {sender} switched the mood to {mood}")
        return True

    matched = _resolve_prompt(sender, message)
    if matched is None:
        return False
    mode, prompt = matched

    if _is_shutup(prompt):
        with _prompt_lock:
            _pending["prompt"] = ""
            _pending["sender"] = sender
            _pending["stop"] = True
        action(f"[AI] {sender} told us to shut up")
        _reset_chatter()
        _close_open_floor()
        return True

    with _prompt_lock:
        _pending["prompt"] = prompt
        _pending["sender"] = sender
        _pending["stop"] = False
        _pending["mode"] = mode
    _note_conversation(sender)
    _reset_chatter()

    action(f"[AI] Captured prompt from {sender}: {prompt}")
    return True


def receiver(sock: socket.socket) -> None:
    buffer = ""
    while True:
        try:
            data = sock.recv(4096).decode("utf-8")
            if not data:
                action("Server closed connection.")
                break
            buffer += data
            lines = buffer.split("\r\n")
            buffer = lines[-1]  # keep incomplete trailing fragment
            for raw in lines[:-1]:
                line = raw.strip()
                if line.startswith("PING "):
                    send(sock, line.replace("PING", "PONG", 1))
                elif _handle_info_line(line):
                    pass  # server information, handled above
                elif " PRIVMSG " in line:
                    parsed = _parse_privmsg(line)
                    if parsed:
                        sender, message = parsed
                        _note_recent(message, sender)
                        if not _handle_ai_prompt(sock, sender, message):
                            irc(f"< {line}")
                            _note_chatter(message)
                elif line:
                    irc(f"< {line}")
        except Exception as e:
            action(f"Receiver error: {e}")
            break


def _take_pending() -> tuple[str, str, bool, str]:
    """Retrieve and clear the pending prompt, sender, stop flag, and mode."""
    with _prompt_lock:
        prompt = _pending["prompt"]
        sender = _pending["sender"]
        stop = _pending["stop"]
        mode = _pending["mode"]
        _pending["prompt"] = ""
        _pending["stop"] = False
        return prompt, sender, stop, mode


def get_pending_prompt() -> str:
    """Retrieve and clear the pending AI prompt."""
    return _take_pending()[0]


def _request_userlist(sock: socket.socket) -> None:
    """Ask the server who is in the channel, right after joining."""
    send(sock, f"WHO {CHANNEL}")


def _system_context(mode: str) -> str:
    """The system prompt for `mode`, with the channel's members woven in.

    Every persona except factual answers *into* the room, so factual answers
    about the world not the people in it, but only factual is context-free
    here: the recent chat history goes into the LLM call as messages (see
    _call_llm), not into the system prompt. The userlist and its mention
    ordering lives in the prompt itself.
    """
    base = _system_prompt(mode)
    if mode == MODE_FACTUAL:
        return base
    parts = [base]
    targets = _mention_targets()
    if targets:
        parts.append(
            "The users in this IRC channel are named: "
            + ", ".join(targets)
            + " Prefer to mention the first one most often (who addressed "
            "you or spoke most recently); then someone who spoke recently; "
            "and only occasionally someone further down the list. Address or "
            "mention users about 50% of the time"
        )
    return "\n\n".join(parts)


def _call_llm(prompt: str, mode: str = MODE_CHAT) -> str:
    """Send prompt to local llama.cpp and return the response text.

    The recent channel lines are injected as messages in the call itself (not
    pasted into the system prompt), so the model sees them as a real chat
    history. This happens on every mode -- factual included -- because every
    reply happens inside an ongoing room.
    """
    system_prompt = _system_context(mode)
    recent = _recent_messages()
    if recent:
        action(f"Injected {len(recent)} lines of chat history as context")
    messages = [
        {"role": "system", "content": system_prompt},
        *recent,
        {"role": "user", "content": prompt},
    ]
    debug(f"System prompt:\n{system_prompt}")
    debug(f"User prompt:\n{prompt}")
    response = _llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        extra_body=LLM_EXTRA_BODY,
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise EmptyLLMReply(
            f"model returned no answer text (finish_reason={choice.finish_reason})"
        )
    return text


# Bytes the wire line spends on framing: "PRIVMSG <chan> :" plus the trailing CRLF.
_IRC_OVERHEAD = len(f"PRIVMSG {CHANNEL} :".encode("utf-8")) + 2


def _truncate_for_irc(text: str) -> str:
    """Truncate text so the resulting PRIVMSG fits in IRC_MAX_LEN bytes."""
    budget = IRC_MAX_LEN - _IRC_OVERHEAD
    if len(text.encode("utf-8")) <= budget:
        return text
    # Cut on a character boundary: shrink until the encoded form fits.
    ellipsis = "…"
    budget -= len(ellipsis.encode("utf-8"))
    cut = text
    while len(cut.encode("utf-8")) > budget:
        cut = cut[:-1]
    return cut + ellipsis


def _format_reply_lines(text: str) -> list[str]:
    """Reflow an LLM reply into at most IRC_MAX_REPLY_LINES sendable lines.

    Words are packed to fill each line up to the byte budget rather than
    following the model's own newlines, so a bulleted answer becomes a few full
    lines instead of one PRIVMSG per bullet. If the reply still does not fit,
    the last line ends in an ellipsis.
    """
    budget = IRC_MAX_LEN - _IRC_OVERHEAD
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate.encode("utf-8")) <= budget:
            current = candidate
            continue
        if current:
            lines.append(current)
        if len(lines) == IRC_MAX_REPLY_LINES:
            # Out of room; mark the last line as truncated.
            lines[-1] = _mark_truncated(lines[-1], budget)
            return lines
        # A single word longer than one line still has to be broken up.
        current = word
        while len(current.encode("utf-8")) > budget:
            lines.append(_truncate_for_irc(current))
            if len(lines) == IRC_MAX_REPLY_LINES:
                lines[-1] = _mark_truncated(lines[-1], budget)
                return lines
            current = current[len(lines[-1]) - 1:]

    if current:
        lines.append(current)
    return lines[:IRC_MAX_REPLY_LINES]


def _mark_truncated(line: str, budget: int) -> str:
    """Append an ellipsis to `line`, shrinking it to stay within `budget` bytes."""
    ellipsis = "…"
    room = budget - len(ellipsis.encode("utf-8"))
    while len(line.encode("utf-8")) > room:
        line = line[:-1]
    return line + ellipsis


def _process_pending(sock: socket.socket) -> None:
    """Check for and respond to any pending AI prompt."""
    prompt, sender, stop, mode = _take_pending()
    if stop:
        send(sock, f"PRIVMSG {CHANNEL} :{SHUTUP_REPLY}")
        _end_conversation()
        return
    if not prompt:
        return

    action(f"[AI] thinking: {prompt}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        reply = _call_llm(prompt, _effective_mode(mode))
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {CHANNEL} :{reply_line}")
        # Compact the reply to one log line; the full multi-line send shows in
        # the IRC log pane above.
        action(f"[AI] {' '.join(reply.split())}")
        # Reading the reply takes time; start their window from now, not from
        # whenever they typed.
        if sender:
            _note_conversation(sender)
    except Exception as e:
        err_msg = f"LLM error: {e}"
        action(f"[AI] error: {err_msg}")
        send(sock, f"PRIVMSG {CHANNEL} :{err_msg}")
    finally:
        with _prompt_lock:
            _busy["on"] = False


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER, PORT))

    send(sock, f"NICK {NICK}")
    send(sock, f"USER {NICK} 0 * :{REALNAME}")

    # Start receiver and wait for server registration to complete
    threading.Thread(target=receiver, args=(sock,), daemon=True).start()
    _registered.wait(timeout=10)

    send(sock, f"JOIN {CHANNEL}")
    _joined["at"] = time.monotonic()
    action(f"[Connected] joined {CHANNEL}")
    _request_userlist(sock)

    # Poll for pending AI prompts, until the TUI asks us to stop.
    try:
        while not _stop_event.is_set():
            _check_silence()
            _process_pending(sock)
            time.sleep(2)
    except KeyboardInterrupt:
        action("[Exiting]")
    finally:
        sock.close()


def status_snapshot() -> dict:
    """A point-in-time read of everything the status pane should show.

    Reads all state under _prompt_lock once, then derives the persona mode and
    the remaining timers. Pure (no side effects) so it can be polled each tick.
    """
    now = time.monotonic()
    with _prompt_lock:
        mood_name = _mood["name"]
        mood_at = _mood["at"]
        recent = len(_recent_lines)
        # Displayed chatter order follows the mention priority (most recently
        # spoken / engaged first), not registration order, so the TUI user list
        # matches the order the names are handed to the LLM. Call the lock-free
        # core here -- status_snapshot already holds _prompt_lock.
        users = _mention_targets_locked()
        floor_open = now < _open_floor["deadline"]
        floor_used = _open_floor["used"]
        chatter = _chatter["count"]
        quiet = now - _activity["at"] if _activity["at"] else 0.0
        busy = _busy["on"]
        convo = _conversation["nick"]
        joined = bool(_joined["at"])
        grace_left = (JOIN_GRACE_PERIOD - (now - _joined["at"])) if joined else 0.0
    mode = MOOD_MODES.get(mood_name, MODE_CHAT)
    mood_left = (MOOD_TIMEOUT - (now - mood_at)) if mood_name != MOOD_BANTER else 0.0
    grace_active = grace_left > 0
    return {
        "mood": mood_name,
        "mode": mode,
        "mood_left": mood_left,
        "history": recent,
        "history_max": RECENT_LINES,
        "users": users,
        "floor_open": floor_open,
        "floor_used": floor_used,
        "floor_left": (_open_floor["deadline"] - now) if floor_open else 0.0,
        "chatter": chatter,
        "quiet": quiet,
        "busy": busy,
        "conversation": convo,
        "grace_active": grace_active,
        "grace_left": grace_left,
        "joined": joined,
    }


if __name__ == "__main__":
    main()
