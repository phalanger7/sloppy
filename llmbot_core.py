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

import profiles
import summarizer


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
warning_sink = _stdout


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


def warning(msg: str) -> None:
    """A problem, shown red in the log pane: a failed or unusable LLM call, a
    rejected summarizer output, or a connection the bot could not make. The
    channel never sees these -- it gets a line in character instead."""
    warning_sink(msg)

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
# The rolling summarizer talks to the same llama-server. Point it at the URL
# configured above so changing the port here is enough -- summarizer.py keeps
# its own default so it still runs stand-alone.
summarizer.API_URL = f"{LLM_BASE_URL}/chat/completions"

# Reasoning models (Qwen3.x and friends) emit a <think> block before the answer.
# llama.cpp routes that into `reasoning_content`, so a budget too small to cover
# it returns finish_reason="length" with an *empty* `content` — the bot then had
# nothing to say. Ask the server to skip thinking, and keep a budget large enough
# to still produce an answer if a template ignores the switch.
LLM_MAX_TOKENS = 512
# How many of the most recent channel lines are kept. This is the buffer, not
# the prompt: the LLM call carries only the last CONTEXT_RECENT_LINES of them
# verbatim (the rolling summary covers the rest), while the full window backs
# the mention ordering.
RECENT_LINES = 200
# How often the background worker wakes to check for a summary trigger.
SUMMARIZE_POLL_INTERVAL = 15
# The worker summarizes when at least this many seconds have passed since the
# last summary, OR more than this many lines have arrived since it -- but only
# once at least SUMMARIZE_MIN_LINES lines have accumulated, so a quiet gap or a
# slow trickle never forces a summary. SUMMARIZE_INTERVAL is the age arm; the
# worker polls far more often so the volume arm fires promptly.
SUMMARIZE_INTERVAL = 600
SUMMARIZE_VOLUME_LINES = 25
SUMMARIZE_MIN_LINES = 5
# Ceiling on the unsummarized-line buffer. Nothing drains it while the bot is
# paused (a paused bot makes no LLM calls), and a server that is down keeps
# handing the lines back to be retried, so without a cap a long pause or a long
# outage would grow it until the eventual request no longer fits the model's
# context. Oldest lines are dropped first: the summary then misses the start of
# the pause, which is the "slightly outdated" this is meant to degrade to.
SUMMARIZE_MAX_PENDING = 200
# How long to wait before retrying after a failed summarizer round-trip. The
# failed lines go back in the buffer, and the age arm is still satisfied, so
# without this the worker would re-attempt on every poll while the server is
# down.
SUMMARIZE_RETRY_AFTER = 60
# The model's rolling summary is rejected (and the previous one kept) if it is
# empty or longer than this many characters, so a runaway model response can
# never overwrite the channel's memory.
SUMMARIZE_MAX_CHARS = 1200
# How many of the most recent channel lines ride along verbatim in the context
# block. The rolling summary covers everything older; this is the sample the
# model needs to answer what was *just* said.
CONTEXT_RECENT_LINES = 20

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
# Skip the join greeting if they left fewer than this many chatlines ago.
GREET_REJOIN_CHATLINES = 5
# Lines shorter than this many characters, or a single word only, are treated
# as noise: not stored in the LLM's recent-history buffer (see _note_recent).
MIN_CHAT_CHARS = 10
# The auto-interject opener waits this long after JOIN so the userlist (and the
# recent channel lines) have time to arrive before the first LLM call.
JOIN_GRACE_PERIOD = 10.0
# How long the poll loop sleeps between passes over the pending work.
POLL_INTERVAL = 2
# How long to wait for the server's 001 Welcome before giving up on a
# connection and reconnecting.
REGISTER_TIMEOUT = 30
# Reconnect backoff. A lost link is retried after RECONNECT_MIN_DELAY and the
# wait doubles up to RECONNECT_MAX_DELAY, so a server that is down, netsplit or
# refusing us is not hammered. A connection that registers and joins resets it.
RECONNECT_MIN_DELAY = 10
RECONNECT_MAX_DELAY = 300
# How often the background worker writes the profile store to disk, when
# anything has changed. Debounced rather than written per line: the receiver
# thread must never wait on a disk write.
PROFILE_SAVE_INTERVAL = 60
# The /props probe answers "is a vision model loaded". That changes only when
# the server is restarted, so it is asked once a minute rather than on every
# poll pass -- it was one HTTP round-trip every two seconds.
VISION_PROBE_INTERVAL = 60
_activity = {"at": 0.0}
# The time the bot joined, so the auto-interject opener can wait
# JOIN_GRACE_PERIOD seconds before it talks (see _within_join_grace). Kept in a
# container -- like _activity -- so main() can record the join without a global
# statement (ruff PLW0603).
_joined = {"at": 0.0}
_open_floor = {"deadline": 0.0, "used": 0}
_chatter = {"count": 0, "last": ""}
# The last 200 channel lines spoken, injected into the LLM call as real chat
# history (see _context_block). A plain parallel buffer keeps the senders in
# lock-step so the mention list can favour recent speakers, not members at
# random. Both stay oldest-first.
_recent_lines = collections.deque(maxlen=RECENT_LINES)
# The nick that spoke each of those lines, in lock-step with _recent_lines, so
# the mention list can favour recent speakers instead of naming members at
# random.
_recent_senders = collections.deque(maxlen=RECENT_LINES)
# Rolling summarizer state (guarded by _prompt_lock): the IRC lines that have
# arrived since the last successful summary, plus the running summary +
# highlights fed into chat prompts. A plain list + the shared _prompt_lock --
# the background worker snapshots and clears it each interval; never a deque.
_pending_summary_lines: list[str] = []
# The running summary + highlights fed into chat prompts, carried forward by
# the summarizer worker. Kept in a container to avoid a global reassignment
# (ruff PLW0603), consistent with the other shared state.
_rolling = {"summary": "", "highlights": []}
# Monotonic time of the last successful summary, for the status pane.
_last_summary_at = {"t": 0.0}
# Monotonic time before which no summarizer retry is attempted, set after a
# failed round-trip so a dead server is not hammered once per poll.
_summary_retry_at = {"t": 0.0}
# Monotonic time of the last /props vision probe, so it runs once a minute
# rather than on every poll pass.
_last_vision_probe = {"t": 0.0}
# What the bot remembers about individual chatters, across sessions. Guarded by
# _prompt_lock like the rest of the shared state -- the store does no locking of
# its own (see profiles.py). _profiles_dirty says whether anything has changed
# since the last write, so an idle channel does not rewrite the file every
# minute.
_profile_store = profiles.ProfileStore()
_profile_path = profiles.default_path()
_profiles_dirty = {"on": False}
_profiles_saved_at = {"t": 0.0}
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
    """The mood to boot into: banter, and the channel can override it.

    Banter is the resting state and never expires, so the bot opens as itself
    rather than a mode someone did not ask for. Factchecking is never the boot
    mood -- booting as a fact-checker nobody asked for is a worse surprise than
    booting funny or booting flat.
    """
    return MOOD_BANTER


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
    # ":<server> 001 <nick> :Welcome" -- matched on the numeric, not on the
    # server's name, so pointing SERVER somewhere else still registers.
    parts = line.split()
    if line.startswith(":") and len(parts) > 1 and parts[1] == "001":
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


# The two privacy commands. Both are anchored or narrow enough that ordinary
# chat does not trip them, and both are gated on the bot being addressed by
# name (see _match_privacy_command).
#
# The forget pattern is anchored at the start of the addressed body and takes
# only a whitelist of filler in front of the verb, so "don't forget about me"
# is not a wipe -- "don't" is not on the list and the anchor keeps the verb
# from matching mid-sentence. "me" is required, so "forget about alice" is not
# a command either.
_FORGET_RE = re.compile(
    r"^(?:(?:please|pls|can|could|would|will|you|hey|ok|okay|now|just|kindly)\s+)*"
    r"forget\s+(?:everything\s+|all\s+|anything\s+|what\s+you\s+know\s+)*"
    r"(?:about\s+)?me\b",
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r"\bwhat\s+(?:do\s+you\s+(?:know|remember)|have\s+you\s+got)\b"
    r"[^?]*\b(?:about|on)\s+me\b",
    re.IGNORECASE,
)


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


def _match_privacy_command(message: str) -> str | None:
    """Return "recall" or "forget" if `message` is a privacy command, else None.

    The bot must be addressed by name (or "AI:"). Neither the follow-up window
    nor the open floor counts: those let ordinary chat through untriggered, and
    "nah forget about me, what about you?" between two humans must never wipe
    somebody's profile.
    """
    text = message.strip()
    if not _bot_is_addressed(text):
        return None
    body = _strip_leading_nick(text)
    if body is None:
        body = _strip_leading_nick(_strip_lead_ins(text))
    if body is None:
        body = _split_prefix(text, "ai:")
    if body is None:
        # Addressed at the end: "what do you know about me, sloppy?"
        trailing = _match_trailing_nick(text)
        body = trailing[1] if trailing else text
    if _FORGET_RE.search(body):
        return "forget"
    if _RECALL_RE.search(body):
        return "recall"
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

    During this window the userlist (353 NAMREPLY) and the recent channel lines
    are still arriving, so every auto-interject trigger is held back until they
    have -- the opener then names real people and reacts to real context.
    """
    with _prompt_lock:
        return time.monotonic() - _joined["at"] < JOIN_GRACE_PERIOD


def _check_silence() -> bool:
    """Break a long silence, then open the floor. Called from the poll loop."""
    with _prompt_lock:
        quiet_for = time.monotonic() - _activity["at"]
        # _busy covers the gap the queue does not: the prompt has been taken
        # off it and the model is mid-generation, so the room is about to hear
        # something and does not also need a "breaking the silence" line.
        busy = bool(_pending["prompt"]) or _pending["stop"] or _busy["on"]
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
# What the channel hears when an LLM call fails. The real error is a red
# warning line in the log pane -- a stack-trace fragment in the channel is
# noise to everyone but whoever is watching the TUI, and it breaks character.
_BRAIN_OFFLINE = [
    "My brain is on hiatus right now.",
    "Gone fishing. Back when the thoughts return.",
    "Don't look at me, I'm just here to watch the scenery.",
    "Nothing upstairs at the moment. Give it a minute.",
    "I appear to have misplaced my train of thought.",
    "Circuits are out to lunch. Ask me again shortly.",
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
    # Somebody who joins after we did is never in a 353/352 reply, so without
    # this they would never make the mention list at all.
    _register_user(nick)
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
    """Record that `nick` left, so a quick rejoin is not greeted.

    They also come off the channel roster: the mention list is who is in the
    room, and _mention_targets_locked already falls back to the last speaker
    for anyone no longer on it.
    """
    if not nick:
        return
    with _prompt_lock:
        _left_at[nick] = _chatlines["count"]
        if nick in _users["names"]:
            _users["names"].remove(nick)
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


def _parse_nick_change(line: str) -> tuple[str, str] | None:
    """Return (old, new) from a NICK line, else None.

    Format: ":old!user@host NICK :new" -- the colon before the new nick is
    optional and servers differ, so both shapes are accepted.
    """
    old, command = _split_event(line)
    if command != "NICK" or not old:
        return None
    _prefix, _sep, rest = line.partition(" NICK ")
    new = rest.strip().lstrip(":").split()[0] if rest.strip() else ""
    return (old, new) if new else None


def _handle_nick_change(old: str, new: str) -> None:
    """Follow somebody through a rename, so they stay one person.

    The profile store links the two names permanently. The bot's own live
    per-nick state is moved across as well, because all of it means "this
    person", not "this string": the roster, the idle and departure clocks, the
    follow-up window, their last image, and the sender labels on the recent
    lines. Those labels are rewritten rather than left alone because they feed
    the mention list, which wants the name to use now -- the log pane has
    already printed the old one, which is the correct history.
    """
    if not old or not new or old.lower() == new.lower():
        return
    with _prompt_lock:
        _profile_store.link(old, new)
        _profiles_dirty["on"] = True
        if old in _users["names"]:
            _users["names"][_users["names"].index(old)] = new
        elif new not in _users["names"]:
            _users["names"].append(new)
        for mapping in (_last_seen, _left_at):
            if old in mapping:
                mapping[new] = mapping.pop(old)
        if _conversation["nick"].lower() == old.lower():
            _conversation["nick"] = new
        image = _recent_images["by_nick"].pop(old.lower(), None)
        if image is not None:
            _recent_images["by_nick"][new.lower()] = image
        for i, sender in enumerate(_recent_senders):
            if sender.lower() == old.lower():
                _recent_senders[i] = new
    action(f"[AI] {old} is now known as {new}")


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
    # Asking to be forgotten, or asking what is stored, is a command about the
    # profile -- not a line to file in it. Computed before the lock; it is two
    # regexes on a string.
    is_privacy_command = _match_privacy_command(message) is not None
    greeting = None
    with _prompt_lock:
        _chatlines["count"] += 1
        if not trivial:
            _recent_lines.append(message.strip())
            _recent_senders.append(sender)
            _pending_summary_lines.append(message.strip())
            # Bounded by hand rather than by a deque: the worker snapshots and
            # clears the whole list, and puts it back when a round-trip fails.
            del _pending_summary_lines[:-SUMMARIZE_MAX_PENDING]
            # The same line, filed under whoever said it. Trivial lines are
            # excluded here for the same reason they are excluded above: "lol"
            # is not something to remember somebody by.
            if not is_privacy_command:
                _profile_store.note_line(sender, message.strip())
                _profiles_dirty["on"] = True
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


def _mention_targets() -> list:
    """Channel nicks ordered for mention priority, most relevant first.

    First the person who addressed the bot, or the last one to speak (the ~70%
    target); then everyone who spoke in the recent-line window, most recent first
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
    spoke in the recent-line window, most recent first (~20% target); then the rest
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

    # Checked before the prompt is resolved, so "sloppy: forget about me" is a
    # command rather than something the model is asked to have an opinion on.
    privacy = _match_privacy_command(message)
    if privacy is not None:
        _handle_privacy_command(sock, sender, privacy)
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
    if command == "NICK" and nick:
        irc(f"< {line}")
        renamed = _parse_nick_change(line)
        if renamed:
            _handle_nick_change(*renamed)
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


def receiver(sock: socket.socket, gone: threading.Event | None = None) -> None:
    """Read the socket until it dies, dispatching each complete line.

    `gone` is the session's disconnect flag: set on the way out so main()'s
    poll loop stops using a socket that is no longer connected and reconnects.
    """
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
            warning(f"Receiver error: {e}")
            break
    # Whatever ended the loop, the link is gone. Wake the poll loop.
    if gone is not None:
        gone.set()


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

    The summarizer's rolling summary, highlights, and a verbatim sample of the
    most recent IRC lines are folded into one system (background/observation)
    message, and the current event rides as the single user message. This
    happens on every mode -- factual included -- because every reply happens
    inside an ongoing room.
    """
    system_prompt = _system_context(mode)
    context_block = _context_block()
    if context_block:
        action("Injected rolling summary + highlights + recent chat as context")
    messages = [
        {"role": "system", "content": system_prompt},
        *context_block,
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
    rolling summary, highlights and recent chat are injected exactly as they
    are for a text reply (see _context_block), because the description is
    spoken into an ongoing room. Uses the shared client, which points at the
    one server that also serves the persona.
    """
    system_prompt = _system_context(MODE_VISION)
    context_block = _context_block()
    if context_block:
        action("Injected rolling summary + highlights + recent chat as context")
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        *context_block,
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


def _probe_vision_if_due() -> None:
    """Run the /props vision probe at most once per VISION_PROBE_INTERVAL.

    Whether an mmproj is loaded changes only when the server is restarted, so
    the poll loop does not need to ask every two seconds.
    """
    now = time.monotonic()
    with _prompt_lock:
        due = now - _last_vision_probe["t"] >= VISION_PROBE_INTERVAL
        if due:
            _last_vision_probe["t"] = now
    if due:
        _probe_vision()


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


def _say_brain_offline(sock: socket.socket, detail: str) -> None:
    """Report a failed LLM call: the real error red in the log pane, and a line
    in character to the channel.

    The channel does not want a Python exception, and printing one there breaks
    the persona for everybody to no purpose -- whoever can fix it is watching
    the TUI, where the full detail goes.
    """
    warning(f"[AI] LLM error on {detail}")
    send(sock, f"PRIVMSG {CHANNEL} :{random.choice(_BRAIN_OFFLINE)}")


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
        _say_brain_offline(sock, f"image request from {sender}: {e}")
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
        _say_brain_offline(sock, f"{prompt!r}: {e}")
    finally:
        with _prompt_lock:
            _busy["on"] = False


def _context_block() -> list:
    """The summarizer's rolling context as ONE system message, or [].

    The rolling summary, highlights, and a verbatim sample of the most recent
    IRC lines are folded into a single system (background/observation) message
    rather than spread across separate user messages, so the model reads them
    as context for the room, not as people talking to it. Each IRC line keeps
    its "sender: text" form -- the sender is the speaker's name inline, never an
    LLM role -- so the recent chat reads as an observation of an external
    conversation. Sections are dropped when empty. Empty until there is
    something to say. Injected in _call_llm ahead of the current user message.
    """
    with _prompt_lock:
        summary = _rolling["summary"].strip()
        highlights = list(_rolling["highlights"])
        senders = list(_recent_senders)
        lines = list(_recent_lines)
    sections = []
    if summary:
        sections.append(f"--- CONVERSATION MEMORY ---\n{summary}")
    if highlights:
        sections.append(
            "--- HIGHLIGHTS ---\n"
            + "\n".join(f"- {h}" for h in highlights)
        )
    recent = [
        (f"{sender}: {text.strip()}" if sender else text.strip())
        for sender, text in zip(senders, lines, strict=False)
    ]
    recent = [r for r in recent if r]
    if recent:
        sections.append(
            "--- RECENT IRC CHAT ---\n"
            + "\n".join(recent[-CONTEXT_RECENT_LINES:])
        )
    if not sections:
        return []
    return [{"role": "system", "content": "\n\n".join(sections)}]


def _reject_reason(summary: object) -> str | None:
    """Why a model-provided summary is unusable, or None if it is valid.

    A usable summary is a non-empty string no longer than SUMMARIZE_MAX_CHARS.
    A whitespace-only string counts as empty.
    """
    if not isinstance(summary, str):
        return "summary was not a string"
    if not summary.strip():
        return "summary was empty"
    if len(summary) > SUMMARIZE_MAX_CHARS:
        return (
            f"summary exceeds {SUMMARIZE_MAX_CHARS} characters ({len(summary)})"
        )
    return None


def _summarize_pending() -> None:
    """Roll the chat summary forward over lines not yet summarized.

    Triggers on age or volume: SUMMARIZE_INTERVAL seconds since the last
    summary, OR SUMMARIZE_VOLUME_LINES lines since it -- but only once at least
    SUMMARIZE_MIN_LINES lines have accumulated, so a quiet gap or a slow trickle
    never forces a summary. Snapshot the unsummarized lines and clear the live
    list first, so the IRC handler can keep appending while the LLM generates --
    the snapshot is an independent list, decoupled from the live one.

    A failed round-trip puts the snapshot back at the front of the buffer, so a
    server that is down costs the channel nothing but time; the summary age is
    left alone (the summary really is still that stale) and a short retry
    window keeps the worker from re-attempting on every poll. Runs off the main
    poll loop and never blocks a reply.
    """
    if _paused["on"]:
        return
    with _prompt_lock:
        if not _pending_summary_lines:
            return
        now = time.monotonic()
        if now < _summary_retry_at["t"]:
            return
        lines_since = len(_pending_summary_lines)
        elapsed = now - _last_summary_at["t"]
        if lines_since < SUMMARIZE_MIN_LINES:
            return
        if not (elapsed >= SUMMARIZE_INTERVAL
                or lines_since >= SUMMARIZE_VOLUME_LINES):
            return
        snapshot = list(_pending_summary_lines)
        _pending_summary_lines.clear()
        summary = _rolling["summary"]
        highlights = list(_rolling["highlights"])
    new_summary, new_highlights, ok = summarizer.summarize_tick_checked(
        summary, highlights, snapshot,
    )
    if not ok:
        # The lines were never summarized. Put them back in front of whatever
        # arrived while the call was in flight (still oldest-first), re-cap the
        # buffer, and leave the rolling state and its age untouched.
        with _prompt_lock:
            _pending_summary_lines[:0] = snapshot
            del _pending_summary_lines[:-SUMMARIZE_MAX_PENDING]
            _summary_retry_at["t"] = time.monotonic() + SUMMARIZE_RETRY_AFTER
        warning(f"SUMMARY FAILED: kept the previous one, retrying "
                f"{len(snapshot)} lines in {SUMMARIZE_RETRY_AFTER}s")
        return
    reject = _reject_reason(new_summary)
    if reject is not None:
        # Discard the unusable summary and keep the previous rolling one, so a
        # model response that is not a string, is empty, or is too large can
        # never overwrite the channel's memory. The warning is posted outside
        # the lock; the highlights are still valid and are applied as usual.
        warning(f"INVALID SUMMARY RECEIVED: {reject}")
        new_summary = summary
        updated = False
    else:
        updated = True
    with _prompt_lock:
        _rolling["summary"] = new_summary
        _rolling["highlights"] = new_highlights
        _last_summary_at["t"] = time.monotonic()
    if updated:
        action(f"[AI] summary updated: {len(new_highlights)} highlights")


def _load_profiles() -> None:
    """Read the profile store off disk, once, before the first connection.

    A missing file is the normal first run. A corrupt one is reported and
    treated as missing: starting with no memory of anybody beats not starting.
    """
    data = profiles.read(_profile_path)
    with _prompt_lock:
        _profile_store.restore(data)
        dropped = _profile_store.prune()
        known = len(_profile_store.known())
        _profiles_dirty["on"] = bool(dropped)
    if data is None:
        action(f"[AI] no profile store at {_profile_path}; starting fresh")
    else:
        action(f"[AI] profiles loaded: {known} people"
               + (f", {dropped} pruned" if dropped else ""))


def _save_profiles_if_due(force: bool = False) -> None:
    """Write the profile store, at most once per PROFILE_SAVE_INTERVAL.

    Skipped entirely when nothing has changed. The snapshot is taken under the
    lock and the disk write happens outside it, so a slow disk never holds up a
    reply. `force` is for shutdown, where the interval does not apply.
    """
    now = time.monotonic()
    with _prompt_lock:
        if not _profiles_dirty["on"]:
            return
        if not force and now - _profiles_saved_at["t"] < PROFILE_SAVE_INTERVAL:
            return
        snapshot = _profile_store.snapshot()
        _profiles_saved_at["t"] = now
        _profiles_dirty["on"] = False
    if not profiles.write(_profile_path, snapshot):
        # The write failed, so the store is still ahead of the file: leave the
        # dirty flag up and let the next tick try again.
        with _prompt_lock:
            _profiles_dirty["on"] = True
        warning(f"[AI] could not save profiles to {_profile_path}")


def _plural(count: int, noun: str) -> str:
    """`1 line` / `4 lines`, for text the channel actually reads."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _fmt_span(seconds: float) -> str:
    """A rough age for a channel line: "3 days", "4 hours", "12 minutes"."""
    for size, name in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {name}{'s' if count != 1 else ''}"
    return "moments"


def _profile_names_locked(profile: dict) -> str:
    """This person's nicks, busiest first. Caller holds _prompt_lock."""
    order = sorted(profile["aliases"], key=lambda a: -profile["aliases"][a])
    return ", ".join(profile["casing"].get(alias, alias) for alias in order)


def _recall_reply(nick: str) -> str:
    """What the bot has on file for `nick`, as one line for the channel.

    Counts, names and dates -- never the stored lines themselves. Reciting
    somebody's own words back into the channel is a worse answer to "what do
    you know about me" than not answering it.
    """
    with _prompt_lock:
        profile = _profile_store.get(nick)
        if profile is None:
            return "Nothing on file for you."
        kept = len(profile["lines"])
        total = profile["line_count"]
        names = _profile_names_locked(profile)
        highlights = len(profile["highlights"])
        age = _fmt_span(time.time() - profile["first_seen"])
    return (
        f"On file for you: {_plural(kept, 'line')} kept of {total} counted, "
        f"first heard {age} ago, under {names}. "
        f"{_plural(highlights, 'highlight')}. "
        "Say 'forget about me' and it all goes."
    )


def _forget_recent_locked(aliases: set[str]) -> None:
    """Drop a person's lines from the short-lived buffers. Caller holds the lock.

    Wiping the profile but leaving their last lines in the recent-chat buffer
    would have the bot quoting somebody it had just promised to forget, in the
    very next reply. The rolling summary is prose and cannot be edited
    surgically -- it ages out instead, and the reply says so rather than
    claiming more than is true.
    """
    kept = [
        (sender, text)
        for sender, text in zip(_recent_senders, _recent_lines, strict=False)
        if sender.lower() not in aliases
    ]
    theirs = {
        text
        for sender, text in zip(_recent_senders, _recent_lines, strict=False)
        if sender.lower() in aliases
    }
    _recent_senders.clear()
    _recent_lines.clear()
    for sender, text in kept:
        _recent_senders.append(sender)
        _recent_lines.append(text)
    _pending_summary_lines[:] = [
        line for line in _pending_summary_lines if line not in theirs
    ]


def _forget_reply(nick: str) -> str:
    """Erase everything on file for `nick` and say what went.

    The store is written to disk immediately rather than waiting for the next
    debounced tick: a wipe that a crash could undo is not a wipe. That write is
    on the receiver thread, which is acceptable for one explicit command on a
    small file.
    """
    with _prompt_lock:
        profile = _profile_store.get(nick)
        if profile is None:
            return "Nothing on file for you to forget."
        total = profile["line_count"]
        names = _profile_names_locked(profile)
        aliases = set(profile["aliases"])
        _profile_store.forget(nick)
        _forget_recent_locked(aliases)
        _profiles_dirty["on"] = True
    _save_profiles_if_due(force=True)
    return (
        f"Forgotten: {_plural(total, 'line')} under {names}, and your recent "
        "chat with them. The rolling channel summary is prose I can't edit "
        "surgically, so anything of yours in there ages out on its own."
    )


def _handle_privacy_command(sock: socket.socket, sender: str, command: str) -> None:
    """Answer a privacy command straight from the receiver thread.

    Templated and immediate, like a mood switch: somebody asking what is stored
    about them, or asking for it to go, wants a straight answer, not the
    persona having a go at it -- and not a two-second wait behind an LLM call.
    """
    reply = _recall_reply(sender) if command == "recall" else _forget_reply(sender)
    for line in _format_reply_lines(reply):
        send(sock, f"PRIVMSG {CHANNEL} :{line}")
    action(f"[AI] {sender} used the '{command}' privacy command")


def _summarize_loop() -> None:
    """Background worker: check for a summary trigger every SUMMARIZE_POLL_INTERVAL.

    Waits out the poll interval on _stop_event so the TUI can stop it promptly;
    runs as a daemon thread started from main(). The actual summary is decided
    inside _summarize_pending, which gates on age/volume and a minimum line
    count. The debounced profile-store write rides along here rather than on
    its own thread: neither job may block a reply, and both are already off the
    poll loop.
    """
    while not _stop_event.wait(SUMMARIZE_POLL_INTERVAL):
        _summarize_pending()
        _save_profiles_if_due()
    # _stop_event was set: flush what the last interval has not written yet.
    _save_profiles_if_due(force=True)


def _connect(gone: threading.Event) -> socket.socket | None:
    """Open one connection: register, join the channel, and ask who is here.

    Returns the live socket, or None when the server could not be reached or
    never completed registration -- the caller backs off and tries again.
    `gone` is this session's disconnect flag, handed to the receiver thread.
    """
    _registered.clear()
    with _prompt_lock:
        # A new connection is a new room, so the old roster goes -- and it goes
        # BEFORE the link exists. Clearing it after registration raced the
        # 353 NAMREPLY that the JOIN below asks for: on a fast server the reply
        # landed first and the clear then wiped the roster it had just filled.
        _users["names"].clear()
    try:
        sock = socket.create_connection((SERVER, PORT), timeout=REGISTER_TIMEOUT)
    except OSError as e:
        warning(f"[Connect] {SERVER}:{PORT} unreachable: {e}")
        return None
    # Blocking from here on: the receiver thread parks in recv() until the
    # server says something or the link dies.
    sock.settimeout(None)
    threading.Thread(target=receiver, args=(sock, gone), daemon=True).start()
    try:
        send(sock, f"NICK {NICK}")
        send(sock, f"USER {NICK} 0 * :{REALNAME}")
        if not _registered.wait(timeout=REGISTER_TIMEOUT):
            raise TimeoutError(f"no 001 Welcome within {REGISTER_TIMEOUT}s")
        send(sock, f"JOIN {CHANNEL}")
        _request_userlist(sock)
    except Exception as e:
        warning(f"[Connect] registration failed: {e}")
        # Closing wakes the receiver thread, which sets `gone` on its way out.
        sock.close()
        return None
    with _prompt_lock:
        # The grace period starts again, so the auto-interject opener waits for
        # the WHO/NAMES replies now on their way.
        _joined["at"] = time.monotonic()
    action(f"[Connected] joined {CHANNEL}")
    return sock


def _run_session(sock: socket.socket, gone: threading.Event) -> None:
    """Poll for pending work until the TUI stops us or the link drops."""
    try:
        while not _stop_event.is_set() and not gone.is_set():
            _check_silence()
            _process_pending(sock)
            _process_pending_vision(sock)
            _probe_vision_if_due()
            time.sleep(POLL_INTERVAL)
    finally:
        sock.close()


def main() -> None:
    """Connect, serve the channel, and reconnect for as long as we are running.

    A link that drops -- a netsplit, a server restart, a connection refused --
    is retried after RECONNECT_MIN_DELAY, doubling to RECONNECT_MAX_DELAY, so a
    server that is down is not hammered; a connection that registers and joins
    resets the backoff. Waits happen on _stop_event, so quitting the TUI does
    not sit through a five-minute backoff.

    The rolling-summarizer worker is started once and survives reconnects: the
    channel's memory is not a property of the socket, and lines that arrived
    before a drop are still worth summarizing after it. The same goes for the
    profile store, which is read once here and written by that worker.
    """
    _load_profiles()
    threading.Thread(target=_summarize_loop, daemon=True).start()
    delay = RECONNECT_MIN_DELAY
    try:
        while not _stop_event.is_set():
            gone = threading.Event()
            sock = _connect(gone)
            if sock is None:
                action(f"[Connect] retrying in {delay}s")
                _stop_event.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)
                continue
            delay = RECONNECT_MIN_DELAY
            _run_session(sock, gone)
            if _stop_event.is_set():
                break
            action(f"[Connect] link lost; reconnecting in {delay}s")
            _stop_event.wait(delay)
    except KeyboardInterrupt:
        pass
    action("[Exiting]")


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
        summary = _rolling["summary"]
        highlights = list(_rolling["highlights"])
        pending = len(_pending_summary_lines)
        last_summary_at = _last_summary_at["t"]
        known_profiles = len(_profile_store.known())
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
        "summary": summary,
        "highlights": len(highlights),
        "highlight_list": highlights,
        "pending_summary": pending,
        "summary_age": (now - last_summary_at) if last_summary_at else 0.0,
        "profiles": known_profiles,
    }


def profiles_snapshot() -> list[dict]:
    """Every profile, most recently seen first, copied for the TUI.

    Separate from status_snapshot because that one is polled every second and
    has no business copying everybody's stored lines. Deep-copied under the
    lock so the UI thread reads a stable picture while the channel talks.
    """
    with _prompt_lock:
        return [
            {
                "nick": p["nick"],
                "aliases": [
                    p["casing"].get(a, a)
                    for a in sorted(p["aliases"], key=lambda a: -p["aliases"][a])
                ],
                "lines": [list(line) for line in p["lines"]],
                "highlights": list(p["highlights"]),
                "quotes": [list(q) for q in p["quotes"]],
                "first_seen": p["first_seen"],
                "last_seen": p["last_seen"],
                "line_count": p["line_count"],
            }
            for p in _profile_store.known()
        ]


if __name__ == "__main__":
    main()
