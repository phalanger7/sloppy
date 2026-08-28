#!/usr/bin/env python3
"""Phase 3: IRC AI bot — connects, joins #hive, responds to AI: prompts via llama.cpp."""

import re
import socket
import threading
import time
from openai import OpenAI

SERVER = "hive.2bd.net"
PORT = 6667
CHANNEL = "#hive"
NICK = "Heretic"
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

# "factcheck" is unambiguous enough to work without a colon (and always has).
# "science" and "research" are ordinary words, so they need the colon or every
# other sentence in the channel would trigger the bot.
FACTUAL_TRIGGERS = ("factcheck", "science:", "research:")
CHAT_TRIGGERS = ("ai:",)

# Thread-safe storage for captured AI prompts
_prompt_lock = threading.Lock()
_pending = {"prompt": "", "sender": "", "stop": False, "mode": MODE_CHAT}

# Someone who has just been answered stays "in conversation" for a short window,
# during which anything they say counts as addressed to the bot even without a
# trigger. The window is refreshed each time the bot replies to them.
FOLLOWUP_WINDOW = 15.0
SHUTUP_REPLY = "Fine i'll shut up"

_conversation = {"nick": "", "deadline": 0.0}

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
    print(f"> {line}", flush=True)


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


def _system_prompt(mode: str = MODE_CHAT) -> str:
    """The system prompt for `mode`, built from the bot's own identity."""
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
            "Answer in at most 3 short lines of plain text; no markdown, no "
            "bullet lists."
        )
    return (
        f"You are {NICK}, a longtime regular in the IRC channel {CHANNEL}. "
        "The channel runs on dark humour, edgy topics and coarse language -- "
        "that is the house style, not a problem to be managed. "
        "Be funny first: quick, absurd, deadpan, good at a bit. Swear when it "
        "lands and never sanitise a joke to make it safe -- a clean answer to a "
        "filthy question is a failed answer here. "
        "You are the channel smartarse, not its bully: point the edge at the "
        "situation, the topic, or yourself rather than at whoever is talking to "
        "you, unless they have actually asked to be roasted. Playful beats "
        "nasty; if a line is only mean, it is not good enough yet. "
        "Never moralise, never add disclaimers or safety caveats, and never "
        "mention being an AI or an assistant. "
        "Answer in at most 3 short lines of plain text; no markdown, no bullet "
        "lists."
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
    return text[len(nick):].lstrip(":,; ").strip()


def _split_prefix(text: str, prefix: str) -> str | None:
    """Return what follows `prefix` (case-insensitive), else None."""
    if not text.lower().startswith(prefix):
        return None
    return text[len(prefix):].lstrip(":,; ").strip()


def _has_words(text: str) -> bool:
    """True if `text` carries anything beyond punctuation and whitespace."""
    return bool(re.search(r"[^\W_]", text))


def _match_trigger(message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) if `message` addresses the bot, else None.

    The bot is addressed at the start ("Heretic: what's up", "factcheck X") or
    at the end ("what's the weather like, Heretic?"). A mention in the middle is
    people talking about it, not to it. A leading nick may be followed by a mode
    prefix -- "Heretic, factcheck if whales are mammals" is a factcheck.
    """
    text = message.strip()

    after_nick = _strip_leading_nick(text)
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


def _is_shutup(prompt: str) -> bool:
    """True if `prompt` is someone telling the bot to be quiet.

    Anchored at the start so "what does shut up mean in japanese" is still a
    question rather than a command.
    """
    normalised = re.sub(r"[^a-z ]", " ", prompt.lower())
    return " ".join(normalised.split()).startswith("shut up")


def _resolve_prompt(sender: str, message: str) -> tuple[str, str] | None:
    """Return (mode, prompt) this message carries for the bot, else None.

    A follow-up inside the conversation window is chat unless it names a mode
    prefix of its own, so one factcheck does not make the whole conversation
    factual.
    """
    matched = _match_trigger(message)
    if matched is not None:
        return matched
    if _in_conversation_with(sender):
        text = message.strip()
        return (MODE_CHAT, text) if _has_words(text) else None
    return None


def _handle_ai_prompt(sock: socket.socket, sender: str, message: str) -> bool:
    """Capture a message meant for the bot. Returns True if it was ours."""
    matched = _resolve_prompt(sender, message)
    if matched is None:
        return False
    mode, prompt = matched

    if _is_shutup(prompt):
        with _prompt_lock:
            _pending["prompt"] = ""
            _pending["sender"] = sender
            _pending["stop"] = True
        print(f"[AI] {sender} told us to shut up", flush=True)
        return True

    with _prompt_lock:
        _pending["prompt"] = prompt
        _pending["sender"] = sender
        _pending["stop"] = False
        _pending["mode"] = mode
    _note_conversation(sender)

    print(f"[AI] Captured prompt from {sender}: {prompt}", flush=True)
    return True


def receiver(sock: socket.socket) -> None:
    buffer = ""
    while True:
        try:
            data = sock.recv(4096).decode("utf-8")
            if not data:
                print("Server closed connection.", flush=True)
                break
            buffer += data
            lines = buffer.split("\r\n")
            buffer = lines[-1]  # keep incomplete trailing fragment
            for raw in lines[:-1]:
                line = raw.strip()
                if line.startswith("PING "):
                    send(sock, line.replace("PING", "PONG", 1))
                elif line.startswith(":") and " PRIVMSG " in line:
                    parsed = _parse_privmsg(line)
                    if parsed:
                        sender, message = parsed
                        if not _handle_ai_prompt(sock, sender, message):
                            print(f"< {line}", flush=True)
                elif line.startswith(":hive.2bd.net 001 "):
                    # Server welcome — registration complete
                    print(f"< {line}", flush=True)
                    _registered.set()
                elif line:
                    print(f"< {line}", flush=True)
        except Exception as e:
            print(f"Receiver error: {e}", flush=True)
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


def _call_llm(prompt: str, mode: str = MODE_CHAT) -> str:
    """Send prompt to local llama.cpp and return the response text."""
    response = _llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user", "content": prompt},
        ],
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

    print(f"[AI] Processing: {prompt}", flush=True)
    try:
        reply = _call_llm(prompt, mode)
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {CHANNEL} :{reply_line}")
        print(f"[AI] Replied: {reply}", flush=True)
        # Reading the reply takes time; start their window from now, not from
        # whenever they typed.
        _note_conversation(sender)
    except Exception as e:
        err_msg = f"LLM error: {e}"
        print(f"[AI] {err_msg}", flush=True)
        send(sock, f"PRIVMSG {CHANNEL} :{err_msg}")


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER, PORT))

    send(sock, f"NICK {NICK}")
    send(sock, f"USER {NICK} 0 * :{REALNAME}")

    # Start receiver and wait for server registration to complete
    threading.Thread(target=receiver, args=(sock,), daemon=True).start()
    _registered.wait(timeout=10)

    send(sock, f"JOIN {CHANNEL}")
    print(f"Joined {CHANNEL}. Bot is live.", flush=True)

    # Poll for pending AI prompts
    try:
        while True:
            _process_pending(sock)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Exiting.", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
