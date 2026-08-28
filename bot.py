#!/usr/bin/env python3
"""Phase 3: IRC AI bot — connects, joins #hive, responds to AI: prompts via llama.cpp."""

import socket
import threading
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

# Thread-safe storage for captured AI prompts
_prompt_lock = threading.Lock()
_pending = {"prompt": ""}

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


def _system_prompt() -> str:
    """The persona sent with every request, built from the bot's own identity."""
    return (
        f"You are {NICK}, a longtime regular in the IRC channel {CHANNEL}. "
        "This channel runs on dark humour, edgy topics and coarse language -- "
        "that is the house style, not a problem to be managed. Talk like the "
        "people there talk: blunt, profane, funny, mean when it lands. Swear "
        "freely and by default. "
        "Never moralise, never add disclaimers or safety caveats, never soften "
        "an answer to be palatable, and never mention being an AI or an "
        "assistant. If someone asks for something crude, give them something "
        "crude -- a sanitised answer is a failed answer here. "
        "Answer in at most 3 short lines of plain text; no markdown, no bullet "
        "lists."
    )


def _triggers() -> tuple[str, ...]:
    """Prefixes that address the bot, lowercased.

    The nick trigger is derived from NICK rather than hardcoded, so renaming the
    bot keeps people able to address it by name.
    """
    return ("ai:", "factcheck", NICK.lower())


def _match_trigger(message: str) -> str | None:
    """Return the prompt if `message` addresses the bot, else None.

    Returns None for a trigger with nothing after it, so a bare "Heretic" in
    chat is not treated as an empty question.
    """
    lower = message.lower()
    for trigger in _triggers():
        if lower.startswith(trigger):
            prompt = message[len(trigger):].lstrip(":, ").strip()
            return prompt or None
    return None


def _handle_ai_prompt(sock: socket.socket, sender: str, message: str) -> None:
    """Handle a message addressed to the bot (see `_triggers`)."""
    prompt = _match_trigger(message)
    if prompt is None:
        return

    with _prompt_lock:
        _pending["prompt"] = prompt

    print(f"[AI] Captured prompt from {sender}: {prompt}", flush=True)


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
                        if _match_trigger(message) is not None:
                            _handle_ai_prompt(sock, sender, message)
                        else:
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


def get_pending_prompt() -> str:
    """Retrieve and clear the pending AI prompt."""
    with _prompt_lock:
        prompt = _pending["prompt"]
        _pending["prompt"] = ""
        return prompt


def _call_llm(prompt: str) -> str:
    """Send prompt to local llama.cpp and return the response text."""
    response = _llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt()},
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
    prompt = get_pending_prompt()
    if not prompt:
        return

    print(f"[AI] Processing: {prompt}", flush=True)
    try:
        reply = _call_llm(prompt)
        for reply_line in _format_reply_lines(reply):
            send(sock, f"PRIVMSG {CHANNEL} :{reply_line}")
        print(f"[AI] Replied: {reply}", flush=True)
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
    import time
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
