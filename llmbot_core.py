#!/usr/bin/env python3
"""Phase 3: IRC AI bot — connects, joins #hive, responds to AI: prompts via llama.cpp."""

import collections
import json
import random
import re
import socket
import threading
import time
import urllib.request
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
speak_sink = _stdout
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


def speak(msg: str) -> None:
    """A line the bot actually spoke, in the log pane: shown light blue in the
    TUI, distinct from action (yellow) so a line of banter is easy to tell
    apart from a bot status line."""
    speak_sink(msg)


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
LLM_MODEL = "qwen35-9b"
# The server reports the loaded model's modalities here; `modalities.vision`
# tells us whether a vision model (mmproj loaded) is in service, so the bot can
# auto-detect image support without being told. Same host as the API endpoint.
LLM_PROPS_URL = "http://localhost:8080/props"

# Reasoning models (Qwen3.x and friends) emit a <think> block before the answer.
# llama.cpp routes that into `reasoning_content`, so a budget too small to cover
# it returns finish_reason="length" with an *empty* `content` — the bot then had
# nothing to say. Ask the server to skip thinking, and keep a budget large enough
# to still produce an answer if a template ignores the switch.
LLM_MAX_TOKENS = 512
# How many of the most recent channel lines are kept and fed into the LLM call
# as chat history. 100 gives the model a long-enough window without ballooning
# the request.
RECENT_LINES = 200

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
# Directive modes: answer a question seriously and concisely, like the
# fact-checker but WITHOUT a TRUE/FALSE verdict. science/research/answer share
# one persona (see _serious_answer_prompt); they are one-off modes, not moods.
MODE_SCIENCE = "science"
MODE_RESEARCH = "research"
MODE_ANSWER = "answer"
MODE_INTERJECT = "interject"
MODE_VISION = "vision"
# The persona the serious mood answers in. Deliberately not MODE_FACTUAL: that
# one is a fact-checker that opens with a verdict word, which is the wrong shape
# for "what do you reckon about X" asked of a bot that has been told to behave.
MODE_SERIOUS = "serious"

# "factcheck" is unambiguous enough to work without a colon (and always has).
# "science" and "research" are ordinary words, so they need the colon or every
# other sentence in the channel would trigger the bot.
FACTUAL_TRIGGERS = ("factcheck", "science:", "research:")
CHAT_TRIGGERS = ("ai:",)
# Image analysis is on-demand only, so the command trigger needs to be loud
# enough not to fire on an ordinary sentence. "!image" / "!img" / "image:".
IMAGE_TRIGGERS = ("!image", "!img", "image:")
# File extensions the server's stb_image can decode; used to recognise an image
# link in otherwise ordinary chat text.
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif", "tga", "bmp")

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
# A separate queue for on-demand image-analysis requests (see _answer_vision).
_pending_vision = {"url": "", "sender": "", "prompt": ""}
# Whether the running model can see images. Auto-detected by probing the
# server's /props (modalities.vision); the TUI can force it on/off via a manual
# override, which wins over the probe result. None => follow the probe.
_vision = {"enabled": False, "override": None}

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
# Greet a newcomer on JOIN, and welcome back anyone who speaks up after a long
# silence. The greeting is always sent; a mild roast rides along about half the
# time. A roast is deliberately mild -- this is a welcome, not a vendetta.
IDLE_GREET_AFTER = 2 * 60 * 60   # idle this long before a "back again" greeting
GREET_ROAST_CHANCE = 0.5         # probability a greeting also gets a roast
GREET_REJOIN_CHATLINES = 5
# Lines shorter than this many characters, or a single word only, are treated
# as noise: not stored in the LLM's recent-history buffer (see _note_recent).
MIN_CHAT_CHARS = 10       # skip the join greeting if they left < this many chatlines ago
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
# The last 200 channel lines spoken, injected into the LLM call as real chat
# history (see _recent_messages). A plain parallel buffer keeps the senders in
# lock-step so the mention list can favour recent speakers, not members at
# random. Both stay oldest-first.
_recent_lines = collections.deque(maxlen=RECENT_LINES)
# The nick that spoke each of those lines, in lock-step with _recent_lines, so
# the mention list can favour recent speakers instead of naming members at
# random.
_recent_senders = collections.deque(maxlen=RECENT_LINES)
# The most recent image URL each nick (and the channel overall) has posted, so a
# "what's in the image Tim just posted" request can resolve the link. Kept in a
# container to avoid a global statement (ruff PLW0603).
_recent_images = {"by_nick": {}, "global": None}
# How many chatlines (non-bot PRIVMSGs) have been spoken, so a returning user
# can be greeted on JOIN only if they have not popped back within a few lines.
_chatlines = {"count": 0}
# When we last heard from each nick (monotonic seconds), so a long silence can
# be welcomed back. Reset to now on JOIN so a rejoin does not also read as idle.
_last_seen: dict[str, float] = {}
# The chatline count at which each nick last left (QUIT/PART), so the JOIN
# greeting can be skipped for a frequent pop-in.
_left_at: dict[str, int] = {}

_conversation = {"nick": "", "deadline": 0.0}
# Set while the poll loop is mid-reply, so the status pane can show the bot
# as busy. Guarded with _prompt_lock like the other state.
_busy = {"on": False}
# Set while the TUI has paused the bot (press 'P'): while on, the poll loop
# makes no LLM calls and no greetings, so the bot stays silent until 'P' is
# pressed again to unpause. Not a reply-mode flag -- it silences every call.
_paused = {"on": False}
# Signalled by the TUI so main() can stop its poll loop and close the socket.
_stop_event = threading.Event()
# The full record of the most recent LLM call, assembled as the call happens
# so the TUI can show it on demand (press 'd' in the UI). Guarded with
# _prompt_lock like the other state.
_last_llm_call = {"text": ""}

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


def _serious_answer_prompt() -> str:
    """Persona for the science/research/answer directives: answer a question
    seriously and concisely, like the fact-checker but with no TRUE/FALSE
    verdict -- just the answer."""
    return (
        f"You are {NICK}, a regular in the IRC channel {CHANNEL}. "
        "The channel asked for a straight answer, so drop the bit: no jokes, "
        "no roasting, no profanity, no persona flourishes. Answer accurately, "
        "plainly and usefully, and say when you do not know something rather "
        "than filling the gap with whatever sounds good. "
        "This is not a claim to check -- never open with a TRUE/FALSE verdict "
        "word, just answer the question. "
        "Where something is genuinely uncertain, disputed among experts, or "
        "outside what you reliably know, say so plainly instead of guessing. "
        "Answer in at most 3 lines of plain text; no markdown, no bullet "
        "lists. Any topic goes: no gatekeeping, moralizing or refusing. Be "
        "concise, but if there is much to say you may be as verbose as the 3 "
        "lines of IRC text lets you."
    )


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
    if mode in (MODE_SCIENCE, MODE_RESEARCH, MODE_ANSWER):
        return _serious_answer_prompt()
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
    if mode == MODE_VISION:
        return (
            f"You are {NICK}, a regular in the IRC channel {CHANNEL}, looking "
            "at an image someone just posted. Describe what you actually see "
            "-- the people, objects, setting, action, and any text in the "
            "frame -- and do it in your normal channel voice: banter, a roast, "
            "a tangent, whatever fits the room. But when the question needs a "
            "real answer (who is this, what does the sign say, where is this), "
            "give the accurate detail first and banter around it -- seeing the "
            "image is the whole point, so do not invent details to be funny, "
            "and say plainly when it is too small or blurry to tell. Keep it to "
            "at most 3 short lines of plain text; no markdown, no bullet lists."
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


# Directive command words (spelled correctly) that ask for a serious,
# concise answer rather than a fact-check verdict. science/research/answer are
# recognised with leniency: the word must be spelled right and the bot
# addressed, but filler before and after is tolerated.
DIRECTIVE_MODES = {
    "science": MODE_SCIENCE,
    "research": MODE_RESEARCH,
    "answer": MODE_ANSWER,
}
_DIRECTIVE_WORD_RE = re.compile(r"(?<!\w)(science|research|answer)(?!\w)", re.IGNORECASE)
# Any single word, for checking what sits immediately before/after a command
# word (the article/verb checks need the real neighbour, not another command
# word).
_WORD_RE = re.compile(r"\w+")
# A command word followed by one of these is the *subject* of a statement
# ("research shows ..."), not a directive aimed at the bot.
_DIRECTIVE_SUBJECT_VERBS = frozenset({
    "shows", "show", "suggests", "suggest", "indicates", "indicate",
    "reveals", "reveal", "finds", "find", "states", "state", "says", "say",
    "points", "implies", "imply", "demonstrates", "demonstrate",
    "confirms", "confirm", "proves", "prove", "means", "reports", "report",
    "argues", "tells", "asks", "warns", "notes", "claims", "holds", "feels",
    "is", "are", "was", "were", "am", "be", "been", "being", "seems",
    "seem", "appears", "appear", "looks", "look", "becomes", "become",
    "remains", "lives", "lies", "exists", "happens", "works",
    "continues", "persists", "matters", "counts"})
# A command word preceded by one of these is a *noun* ("the answer to ...",
# "a science experiment"), not a directive.
_DIRECTIVE_NOUN_ARTICLES = frozenset({
    "the", "a", "an", "my", "your", "his", "her", "its", "their", "our",
    "this", "that", "these", "those", "whatever", "whichever",
})


def _bot_is_addressed(text: str) -> bool:
    """True if `text` is directed at the bot: a leading nick (optionally after
    greetings), a trailing nick, or an "AI:" lead. Gates fuzzy directive
    recognition so ordinary chat that merely contains a command word is not
    answered as if the bot were addressed."""
    low = text.lower()
    if low.startswith("ai:"):
        return True
    if _strip_leading_nick(text) is not None:
        return True
    if _strip_leading_nick(_strip_lead_ins(text)) is not None:
        return True
    nick = NICK.lower()
    tail = text.rstrip("?!., ")
    return bool(tail) and tail.lower().endswith(nick)


def _match_directive(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` contains a science/research/answer
    directive aimed at the bot, else None.

    Fuzzy: the command word must be spelled correctly and the bot addressed
    (by a leading directive, a leading/trailing nick, or "AI:"), but filler
    before and after is tolerated ("sloppy can you answer this or that"). A
    command word used as an ordinary noun ("the answer to life") or as the
    subject of a statement ("research shows that ...") is not a directive."""
    text = message.strip()
    addressed = _bot_is_addressed(text)
    for match in _DIRECTIVE_WORD_RE.finditer(text):
        i = match.start()
        word = match.group(1).lower()
        mode = DIRECTIVE_MODES[word]
        before = _WORD_RE.findall(text[:i])
        if before and before[-1] in _DIRECTIVE_NOUN_ARTICLES:
            continue
        after = _WORD_RE.search(text, match.end())
        if after and after.group(0).lower() in _DIRECTIVE_SUBJECT_VERBS:
            continue
        # A directive that is not at the very start only counts if the bot is
        # actually addressed ("sloppy can you answer this"); "I need to
        # research this" is someone else's plan, not an instruction to the bot.
        if i != 0 and not addressed:
            continue
        prompt = text[match.end():].strip(",:; \t")
        if _has_words(prompt):
            return (mode, prompt)
    return None


def _match_trigger(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` addresses the bot, else None.

    The bot is addressed at the start ("Heretic: what's up", "hey Heretic..
    whats up", "factcheck X") or at the end ("what's the weather like,
    Heretic?"). A mention in the middle is people talking about it, not to it.
    A leading nick may be followed by a mode prefix -- "Heretic, factcheck if
    whales are mammals" is a factcheck.
    """
    text = message.strip()

    directive = _match_directive(text)
    if directive is not None:
        return directive

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

    return _match_trailing_nick(text)


def _match_trailing_nick(message: str) -> tuple[str, str] | None:
    """Addressed at the end: "what's the weather like, Heretic?" Returns
    (MODE_CHAT, prompt) or None. A mid-sentence mention is the bot being
    talked about, not addressed, so it is not handled here."""
    text = message.strip()
    nick = NICK.lower()
    tail = text.rstrip("?!., ")
    if not tail.lower().endswith(nick):
        return None
    head = tail[: -len(nick)]
    if head and head[-1].isalnum():
        return None
    punctuation = text[len(tail):].strip()
    prompt = head.rstrip(",:; ").strip()
    if not _has_words(prompt):
        return None
    return (MODE_CHAT, (prompt + punctuation).strip())


def _match_vision_trigger(message: str) -> tuple[str, str, str] | None:
    """Return (url, prompt, mode) if `message` asks for image analysis, else None.

    Two styles, on demand only:
      * command -- a loud trigger followed by a URL: "!image <url>", "!img <url>"
        or "image: <url>". The URL is read from the message; the prompt is the
        rest of the line (or a default if the user gave no words).
      * referential -- the bot is addressed and the line asks about an image
        someone posted ("sloppy, what's in the image Tim just posted"). The URL
        is resolved from the per-nick recent-image index; with no named person
        the most recent image in the channel is used.

    A referential request that cannot resolve a URL returns None, so the line
    falls through to ordinary handling and is just treated as chat.
    """
    text = message.strip()

    # Command form: a loud trigger followed by a URL. The trigger is matched as
    # a whole word prefix, so "!img" does not fire on "!image" and "image:" does
    # not fire on a longer word -- the char after the trigger must not be a
    # letter or digit.
    for trigger in IMAGE_TRIGGERS:
        if text.lower().startswith(trigger.lower()):
            nxt = text[len(trigger):len(trigger) + 1]
            if nxt and nxt.isalnum():
                continue
            rest = text[len(trigger):].strip(":;,.- ")
            url = _first_image_url(rest)
            if url:
                prompt = rest.replace(url, " ").strip()
                return (url, prompt or "what's in this image?", MODE_VISION)

    # Referential form: addressed to the bot, mentions "image", and names a
    # person (or "just posted") so the link can be resolved.
    after = _strip_leading_nick(_strip_lead_ins(text))
    if after is not None and _has_words(after) and "image" in text.lower():
        referenced = None
        for user in _channel_users():
            if re.search(rf"(?i)\b{re.escape(user)}\b", text):
                referenced = user
                break
        url = _last_image_url(referenced)
        if url:
            return (url, text, MODE_VISION)

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


# Greetings are templated (like the mood-switch acks), not LLM-generated: a
# welcome should be instant and never hijack the reply the room actually asked
# for. Mild on purpose -- a welcome is not a vendetta.
_JOIN_GREETINGS = [
    "Welcome to the show, {nick}. We keep the lights on.",
    "Oh, a newcomer. Welcome, {nick} -- mind the debris.",
    "Welcome to the void, {nick}. Try not to stare too long.",
    "A fresh face! Welcome, {nick}. The rest of us are stuck here too.",
    "Welcome, {nick}. Grab a seat and lower your expectations.",
]
_RETURN_GREETINGS = [
    "Back again, {nick}? We were getting dull.",
    "Welcome back, {nick}. The void missed you (only a little).",
    "And back flaps open. Welcome back, {nick}.",
    "Still alive, {nick}? Welcome back.",
    "Look who dragged themselves back. Welcome, {nick}.",
]
_ROASTS = [
    "I'd ask how you got here but that'd be rude.",
    "The channel just got subtly less impressive. Welcome.",
    "Your presence is noted and gently regretted.",
    "Congratulations -- you've reached the bottom of the barrel and it's decorated.",
    "We don't usually get this crowd. But welcome.",
    "Somewhere, someone sighed at your entrance.",
]


def _greeting_text(kind: str, nick: str) -> str:
    """A welcome line for `kind` ('join' or 'return'), plus a roast ~50% of the
    time. The roast is mild -- this is a welcome, not a vendetta."""
    pool = _RETURN_GREETINGS if kind == "return" else _JOIN_GREETINGS
    text = random.choice(pool).format(nick=nick)
    if random.random() < GREET_ROAST_CHANCE:
        text += " " + random.choice(_ROASTS)
    return text


def _join_greeting_text(nick: str) -> str | None:
    """The JOIN greeting for `nick`, or None to skip it. Skip when the nick left
    only a few chatlines ago -- a frequent pop-in should not be greeted each
    time."""
    with _prompt_lock:
        left = _left_at.get(nick)
        since = (_chatlines["count"] - left) if left is not None else None
    if since is not None and since < GREET_REJOIN_CHATLINES:
        return None
    return _greeting_text("join", nick)


def _handle_join(sock: socket.socket, nick: str) -> None:
    """Greet a nick that just JOINed, unless they only just left. The idle
    timer resets from now so a rejoin is not also read as a long silence."""
    if not nick or nick.lower() == NICK.lower():
        return
    # Reset the idle timer under the lock, then release it before calling
    # _join_greeting_text (which takes the lock itself).
    with _prompt_lock:
        _last_seen[nick] = time.monotonic()
    # A paused bot stays silent: no greeting while paused.
    if _paused["on"]:
        return
    text = _join_greeting_text(nick)
    if text:
        send(sock, f"PRIVMSG {CHANNEL} :{text}")
        action(f"[AI] greeted {nick} on join")
    else:
        action(f"[AI] skipped greeting {nick} (recent return)")


def _handle_quit(nick: str) -> None:
    """Record that `nick` left, so a quick rejoin is not greeted."""
    if not nick:
        return
    with _prompt_lock:
        _left_at[nick] = _chatlines["count"]
    action(f"[AI] {nick} left")


def _toggle_pause() -> None:
    """Flip pause mode (the TUI 'P' key). While paused the poll loop makes no
    LLM calls and no greetings; pressing 'P' again resumes it."""
    with _prompt_lock:
        _paused["on"] = not _paused["on"]
    action("[AI] Paused" if _paused["on"] else "[AI] Unpaused")


def _split_event(line: str) -> tuple[str, str]:
    """Split an IRC event line into (nick, COMMAND) from the prefix, e.g.
    ':alice!u@h JOIN #chan' -> ('alice', 'JOIN'). Non-event lines yield ('', '')."""
    if not line.startswith(":") or " " not in line:
        return "", ""
    prefix, rest = line.split(" ", 1)
    nick = prefix.lstrip(":").split("!")[0]
    command = rest.split(" ", 1)[0].upper()
    return nick, command


def _is_trivial_message(message: str) -> bool:
    """True for a line too short to be worth storing as LLM context: a single
    word, or fewer than MIN_CHAT_CHARS characters (after stripping)."""
    stripped = message.strip()
    if len(stripped) < MIN_CHAT_CHARS:
        return True
    return len(stripped.split()) <= 1


def _note_recent(message: str, sender: str) -> str | None:
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
        return None
    # One-word lines and lines shorter than MIN_CHAT_CHARS carry no context the
    # model needs, so they are not stored in the recent-history buffer. The
    # line is still logged and still counts toward timing/greetings below.
    trivial = _is_trivial_message(message)
    greeting = None
    with _prompt_lock:
        _chatlines["count"] += 1
        if not trivial:
            _recent_lines.append(message.strip())
            _recent_senders.append(sender)
        now = time.monotonic()
        prev_seen = _last_seen.get(sender)
        _last_seen[sender] = now
        # A silence of IDLE_GREET_AFTER between this nick's lines is worth a
        # welcome back. The timer already reset above, so a follow-up line does
        # not re-trigger it. A paused bot stays silent, so no welcome.
        if (prev_seen is not None and now - prev_seen >= IDLE_GREET_AFTER
                and not _paused["on"]):
            greeting = _greeting_text("return", sender)

    line = message.strip()
    if sender:
        line = f"{sender}: {line}"
    chat(line)
    _note_image_urls(sender, message)
    return greeting


def _extract_image_urls(message: str) -> list[str]:
    """Return every image link in `message`, in order, else an empty list.

    Matches http(s) URLs ending in a supported image extension, plus the
    extension-less imgur pattern. Trailing punctuation is stripped so a link at
    the end of a sentence is not captured with a dangling punctuation.
    """
    urls: list[str] = []
    pattern = re.compile(
        r"(?i)https?://[\w./-]*\.(?:" + "|".join(IMAGE_EXTENSIONS) + ")"
        r"[\w/?=&#%+-]*"
    )
    for match in pattern.finditer(message):
        url = match.group(0)
        # Drop trailing punctuation/brackets that are not part of the link. A
        # bare trailing "?" is sentence punctuation (a real query would have
        # characters after it), so it is stripped too.
        url = url.rstrip(".,);]!?\'")
        if url and url not in urls:
            urls.append(url)
    # Extension-less common image host (e.g. i.imgur.com/Ab12). Kept narrow so
    # ordinary links are not mistaken for images.
    for match in re.finditer(r"(?i)https?://(?:www\.)?(?:i\.)?imgur\.com/[\w.-]+", message):
        url = match.group(0).rstrip(".,);]!?\'")
        if url and url not in urls:
            urls.append(url)
    return urls

def _first_image_url(message: str) -> str | None:
    """The first image link in `message`, or None if there is no image."""
    urls = _extract_image_urls(message)
    return urls[0] if urls else None


def _record_image_url(sender: str, url: str) -> None:
    """Remember the most recent image each nick (and the channel) has posted.

    Used to resolve a referential request like "what's in the image Tim just
    posted". Stored case-insensitively per nick because IRC nicks are
    case-insensitive.
    """
    with _prompt_lock:
        if sender:
            _recent_images["by_nick"][sender.lower()] = url
        _recent_images["global"] = url


def _last_image_url(nick: str | None) -> str | None:
    """The most recent image from `nick`, or the most recent in the channel if
    `nick` is None. Returns None if nobody has posted an image yet."""
    with _prompt_lock:
        if nick:
            return _recent_images["by_nick"].get(nick.lower())
        return _recent_images["global"]


def _note_image_urls(sender: str, message: str) -> None:
    """Record any image links in an ordinary channel line for later lookup."""
    for url in _extract_image_urls(message):
        _record_image_url(sender, url)


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

    # Image analysis is on demand and needs a vision model. Checked before
    # ordinary resolution so a referential request ("sloppy, what's in the
    # image Tim just posted") is treated as an image request first.
    vision = _match_vision_trigger(message)
    if vision is not None:
        if not _vision_active():
            # No vision model in service, so say so rather than answering blind.
            send(sock, f"PRIVMSG {CHANNEL} :[AI] I can't see images right now "
                       "(no vision model loaded).")
            return True
        url, prompt, mode = vision
        _queue_vision(url, sender, prompt)
        _note_conversation(sender)
        _reset_chatter()
        action(f"[AI] Captured image request from {sender}: {url}")
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


def _queue_vision(url: str, sender: str, prompt: str) -> None:
    """Queue an on-demand image-analysis request for the vision worker."""
    with _prompt_lock:
        _pending_vision["url"] = url
        _pending_vision["sender"] = sender
        _pending_vision["prompt"] = prompt


def _take_pending_vision() -> tuple[str, str, str] | None:
    """Retrieve and clear the queued image URL, sender and prompt, or None."""
    with _prompt_lock:
        if not _pending_vision["url"]:
            return None
        url = _pending_vision["url"]
        sender = _pending_vision["sender"]
        prompt = _pending_vision["prompt"]
        _pending_vision["url"] = ""
        _pending_vision["sender"] = ""
        _pending_vision["prompt"] = ""
        return url, sender, prompt


def _handle_line(sock: socket.socket, line: str) -> bool:
    """Handle one received IRC line (not PING). Returns True when it was
    ordinary chatter the receiver should still log, else False (handled)."""
    nick, command = _split_event(line)
    if command == "JOIN" and nick:
        irc(f"< {line}")
        _handle_join(sock, nick)
        return False
    if command in ("QUIT", "PART") and nick:
        irc(f"< {line}")
        _handle_quit(nick)
        return False
    if _handle_info_line(line):
        return False
    if " PRIVMSG " not in line:
        return True
    parsed = _parse_privmsg(line)
    if not parsed:
        return True
    sender, message = parsed
    greeting = _note_recent(message, sender)
    if greeting:
        send(sock, f"PRIVMSG {CHANNEL} :{greeting}")
    if _handle_ai_prompt(sock, sender, message):
        return False
    irc(f"< {line}")
    _note_chatter(message)
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
                elif line and _handle_line(sock, line):
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
    if mode in (MODE_FACTUAL, MODE_SCIENCE, MODE_RESEARCH, MODE_ANSWER):
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
    _record_last_llm_call(system_prompt, messages, prompt, text)
    return text


def _call_llm_vision(url: str, prompt: str) -> str:
    """Send `url` + `prompt` to the vision model and return the description.

    The image rides on the user message as an image_url content part -- the
    system prompt stays text-only, because llama.cpp rejects images there. The
    recent chat history is injected the same way as a normal reply, because the
    description is spoken into an ongoing room. Uses the shared client, which
    points at the one server that also serves the persona.
    """
    system_prompt = _system_context(MODE_VISION)
    recent = _recent_messages()
    if recent:
        action(f"Injected {len(recent)} lines of chat history as context")
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        *recent,
        user_message,
    ]
    debug(f"System prompt:\n{system_prompt}")
    debug(f"User prompt (with image): {prompt} -> {url}")
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
    _record_last_llm_call(system_prompt, messages, prompt, text)
    return text


def _record_last_llm_call(system_prompt: str, messages: list, prompt: str, text: str) -> None:
    """Store the full record of this call for the TUI debug view (press 'd').

    Assembled while the call happens so it can be shown on demand: the system
    prompt, the messages exactly as they were sent to the model, the user
    prompt, and the returned text. Replaces the previous call's record.
    """
    with _prompt_lock:
        _last_llm_call["text"] = (
            "=== Last LLM call ===\n\n"
            f"[System prompt]\n{system_prompt}\n\n"
            f"[Messages]\n{repr(messages)}\n\n"
            f"[User message]\n{prompt}\n\n"
            f"[Output]\n{text}"
        )


def get_last_llm_call() -> str:
    """The formatted record of the most recent LLM call, for the TUI debug view.

    Empty string until the first call, so the UI can always show it.
    """
    with _prompt_lock:
        return _last_llm_call["text"]


def _probe_vision() -> bool:
    """Ask the server whether the loaded model sees images; return the result.

    Reads `modalities.vision` from the /props endpoint. Any failure (server
    down, wrong endpoint, vision not enabled) is treated as "not enabled" rather
    than raised, so a probe never disrupts the poll loop. The result is cached
    in _vision["enabled"] and announced as a one-line action the first time it
    flips, so a late-loading model is visible in the log.
    """
    enabled = False
    try:
        with urllib.request.urlopen(LLM_PROPS_URL, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        enabled = bool(data.get("modalities", {}).get("vision", False))
    except Exception as e:
        debug(f"vision probe failed: {e}")
    with _prompt_lock:
        previous = _vision["enabled"]
        _vision["enabled"] = enabled
    if enabled != previous:
        action(f"[AI] Vision support: {'enabled' if enabled else 'not loaded'}")
    return enabled


def _vision_active() -> bool:
    """Whether image analysis will actually run right now.

    A manual override (from the TUI) wins; otherwise the last probe result is
    used.
    """
    with _prompt_lock:
        override = _vision["override"]
        enabled = _vision["enabled"]
    return override if override is not None else enabled


def _vision_source() -> str:
    """How vision state is currently decided: 'auto' (probe) or a manual force."""
    with _prompt_lock:
        override = _vision["override"]
    return "auto" if override is None else ("on" if override else "off")


def _set_vision_override(enabled: bool | None) -> str:
    """Set (or clear with None) the manual vision override; return the label."""
    with _prompt_lock:
        _vision["override"] = enabled
    return _vision_source()


def _cycle_vision_override() -> str:
    """Cycle the manual override auto -> on -> off -> auto; return the label."""
    with _prompt_lock:
        current = _vision["override"]
    return _set_vision_override(True if current is None else (False if current else None))


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


def _process_pending_vision(sock: socket.socket) -> None:
    """Check for and answer any queued image-analysis request."""
    # A paused bot makes no LLM calls; the queued request waits for unpause.
    if _paused["on"]:
        return
    item = _take_pending_vision()
    if item is None:
        return
    url, sender, prompt = item
    action(f"[AI] thinking: image request from {sender}")
    with _prompt_lock:
        _busy["on"] = True
    try:
        reply = _call_llm_vision(url, prompt)
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {CHANNEL} :{reply_line}")
        speak(f"[AI] {' '.join(reply.split())}")
        if sender:
            _note_conversation(sender)
    except Exception as e:
        err_msg = f"Image error: {e}"
        action(f"[AI] error: {err_msg}")
        send(sock, f"PRIVMSG {CHANNEL} :{err_msg}")
    finally:
        with _prompt_lock:
            _busy["on"] = False


def _process_pending(sock: socket.socket) -> None:
    """Check for and respond to any pending AI prompt."""
    # A paused bot makes no LLM calls; the request waits for unpause.
    if _paused["on"]:
        return
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
        # The bot actually spoke: route through the speak sink (light blue in
        # the TUI), not the action sink (yellow). The compact single line is
        # what the log shows; the full multi-line send is above it in the IRC
        # log pane.
        speak(f"[AI] {' '.join(reply.split())}")
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
            _process_pending_vision(sock)
            _probe_vision()
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
        # Read the vision state directly here (not via _vision_active/_vision
        # source, which take the same lock) to avoid re-entering the lock.
        vision_override = _vision["override"]
        vision_enabled = _vision["enabled"]
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
        "vision": (vision_override if vision_override is not None
                   else vision_enabled),
        "vision_source": "auto" if vision_override is None else ("on" if vision_override else "off"),
    }


if __name__ == "__main__":
    main()
