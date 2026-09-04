#!/usr/bin/env python3
"""IRC log summarizer.

A tiny, dependency-light rolling summarizer for an IRC channel. Given the
previous rolling ``(summary, highlights)`` state and a batch of new chat lines,
it asks a local llama-server (an OpenAI-compatible ``/v1/chat/completions``
endpoint) for an updated summary and highlights in a fixed JSON shape, so the
bot can keep a persistent sense of "what's been going on" without re-feeding the
whole channel history on every reply.

Public entry point: :func:`summarize_tick`. It never raises -- on any failure
it returns its inputs unchanged, so a caller can keep using the prior rolling
state and degrade gracefully.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import requests

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
REQUEST_TIMEOUT = 120
MAX_HIGHLIGHTS = 5

SYSTEM_PROMPT = """\
You maintain a rolling memory of an IRC channel.

Your job is to preserve information that will help an IRC bot understand what
has been happening and participate naturally in future conversation.

Return ONLY valid JSON matching the requested schema.

SUMMARY:
Write a concise rolling summary of the conversation, approximately 400–700
characters in length.

Prioritize:
- ongoing topics and discussions
- unresolved questions, problems, or disagreements
- things people are working on, planning, or trying to accomplish
- important facts established in the conversation
- notable events or developments
- changes of topic that may provide useful context later
- recurring jokes, references, or conversational context that may matter later

Do not waste space on greetings, filler, trivial exchanges, or merely describing
that people were talking. Record what they actually discussed.

The summary is a ROLLING summary: incorporate useful information from the
previous summary and the new log lines. Preserve important ongoing context
even when it was mentioned earlier, while removing information that is no
longer useful or has become stale.

HIGHLIGHTS:
Return up to 5 memorable and useful pieces of information.

Carry forward previous highlights that are still useful, replace stale
highlights when better information appears, and add genuinely important new
events or facts.

Prefer:
- durable conversational facts
- significant developments
- useful technical or project details
- running jokes or references that may matter later
- genuinely memorable events or notable quotes

Do not create highlights merely to fill the list. Fewer than 5 is fine.

A highlight must contain enough context to remain useful without the original
IRC lines.

ACCURACY:
Do not invent information, attribute statements to people who did not make
them, or infer facts that were not stated or clearly supported by the log.

The previous summary and highlights are context, not facts that must be
preserved if the new conversation contradicts them. Prefer the most recent
explicit information.

Output exactly this JSON structure:
{
  "summary": "...",
  "highlights": ["...", "..."]
}"""


def _previous_highlights_text(highlights: list[str]) -> str:
    """Render previous highlights as a bullet list (empty string if none)."""
    return "\n".join(f"- {h.strip()}" for h in highlights or [] if h and h.strip())


def _clean_highlights(raw: Any) -> list[str]:
    """Normalise model-provided highlights: strings only, stripped, capped.

    De-duplicates while preserving order and stops at :data:`MAX_HIGHLIGHTS`, so
    a chatty model cannot inflate the rolling highlight list beyond the cap.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= MAX_HIGHLIGHTS:
            break
    return out


def summarize_tick(
    previous_summary: str,
    previous_highlights: list[str],
    new_lines: list[str],
) -> tuple[str, list[str]]:
    """Roll the channel summary forward over ``new_lines``.

    Sends the previous rolling state plus the new IRC log lines to llama-server
    and returns the updated ``(summary, highlights)``.

    On any failure -- empty ``new_lines``, a network/HTTP error, a malformed
    response, or bad JSON -- returns the inputs unchanged, so a caller can keep
    using the prior rolling state. Never raises out of this function.
    """
    # No new work: keep the rolling state exactly as it was, and skip the
    # server round-trip entirely.
    if not new_lines:
        return previous_summary, previous_highlights

    new_block = "\n".join(new_lines)
    user_message = (
        f"Previous summary:\n{previous_summary}\n\n"
        f"Previous highlights:\n{_previous_highlights_text(previous_highlights)}\n\n"
        f"New log lines:\n{new_block}"
    )

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "highlights": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "highlights"],
                },
            },
        },
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("JSON top-level value is not an object")
        if "summary" not in data or "highlights" not in data:
            raise ValueError("JSON missing 'summary' or 'highlights' key")

        summary = str(data["summary"]).strip()
        highlights = _clean_highlights(data["highlights"])
        return summary, highlights
    except Exception as exc:  # noqa: BLE001 - contract: never raise out of here
        print(f"summarize_tick: failed: {exc}", file=sys.stderr)
        return previous_summary, previous_highlights


if __name__ == "__main__":
    # Standalone smoke test against a running llama-server. With no server it
    # exercises the failure path and prints the inputs unchanged to stderr.
    demo_lines = [
        "Alice: morning everyone",
        "Bob: did anyone see the patch Tim pushed?",
        "Tim: yeah it fixes the join race",
    ]
    result = summarize_tick("Initial summary is empty.", [], demo_lines)
    print("Result:", result)
