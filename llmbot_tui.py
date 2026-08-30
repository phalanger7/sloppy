#!/usr/bin/env python3
"""Textual front-end for the IRC AI bot.

Imports llmbot_core (the business logic, a fork of bot.py) and paints two
panes:

  * top-left -> the log          every action the bot takes (being addressed,
                   replying, switching mood, interjecting, injecting its chat
                   history, shutting down) and every line of channel chat in
                   "nick: text" format. Actions are bold and bright; plain chat
                   lines are not.
  * top-right -> status          (mood/mode, chat-history buffer count,
                   open-floor state, chatter count, reply/quiet/join timers).

The raw IRC log is intentionally not shown — it is noise, not bot state. The
bot runs in a background thread; the module sinks are wired to the app so every
meaningful line is posted back to the UI thread. Run with:

    python3 llmbot_tui.py

Only the Python standard library's curses-free Textual is used, and it is
already installed in the system Python that runs this bot.
"""

from __future__ import annotations

import threading

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from rich.text import Text
from textual.widgets import RichLog, Static

import llmbot_core as bot


class LogLine(Message):
    """A line for the log pane, posted from the background thread."""

    def __init__(self, text: str, *, action: bool) -> None:
        super().__init__()
        self.data = text
        self.is_action = action


def _fmt_duration(seconds: float) -> str:
    """`125.0` -> `2m05s`; sub-60 stays in seconds."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _format_status(snap: dict) -> str:
    """Render a status snapshot into a single-column, monospace block."""
    mode_left = _fmt_duration(snap["mood_left"]) if snap["mood"] != "banter" else "—"
    floor = "active" if snap["floor_open"] else "inactive"
    if snap["floor_open"]:
        floor += f" ({snap['floor_used']}/{bot.OPEN_FLOOR_MAX_PROMPTS}, " \
                 f"{_fmt_duration(snap['floor_left'])} left)"
    quiet = _fmt_duration(snap["quiet"])
    quiet += " idle" if snap["quiet"] >= bot.SILENCE_TIMEOUT else ""
    if snap["grace_active"]:
        grace = f"{_fmt_duration(snap['grace_left'])} grace"
    elif snap["joined"]:
        grace = "up"
    else:
        grace = "connecting"
    busy = "replying" if snap["busy"] else "idle"
    users = ", ".join(snap["users"]) if snap["users"] else "(none yet)"
    mode_note = f" ({snap['mode']} persona)" if snap["mode"] != "chat" else ""
    lines = [
        f"Mood / Mode : {snap['mood']}{mode_note}",
        f"Mode left   : {mode_left}",
        f"Chat history: {snap['history']}/{snap['history_max']}",
        f"Open floor  : {floor}",
        f"Chatter     : {snap['chatter']}/{bot.IDLE_INTERJECT_AFTER} to interject",
        f"Quiet       : {quiet}",
        f"Users       : {len(snap['users'])} — {users}",
        f"Join        : {grace}",
        f"Bot         : {busy}",
    ]
    return "\n".join(lines)


class LLMBotApp(App[None]):
    """The bot's TUI."""

    CSS = """
    #log {
        width: 62%;
        border-right: thick $error;
        background: $panel;
        color: $text;
    }
    #status {
        width: 38%;
        border-left: thick $warning;
        padding: 0 1 0 1;
        align: left top;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            RichLog(id="log", auto_scroll=True, highlight=False),
            Static(id="status"),
        )

    def on_mount(self) -> None:
        # Route the core's meaningful lines into the log via thread-safe
        # messages. The raw IRC sink is dropped — it is noise, not bot state.
        bot.irc_sink = lambda _text: None
        bot.action_sink = self._on_action
        bot.chat_sink = self._on_chat
        bot.debug_sink = lambda _text: None
        self._bot_thread = threading.Thread(target=bot.main, daemon=True)
        self._bot_thread.start()
        self.set_interval(1.0, self._refresh_status)
        self._refresh_status()

    def _on_action(self, text: str) -> None:
        self.post_message(LogLine(text, action=True))

    def _on_chat(self, text: str) -> None:
        self.post_message(LogLine(text, action=False))

    def on_log_line(self, msg: LogLine) -> None:
        log = self.query_one("#log", RichLog)
        if msg.is_action:
            # Bold + bright so the bot's own actions stand out from chat.
            # A Text object (not a markup string) keeps brackets in lines
            # like "[AI] ..." literal instead of being parsed as tags.
            log.write(Text(msg.data, style="bold bright_yellow"))
        else:
            log.write(Text(msg.data))

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(_format_status(bot.status_snapshot()))

    def on_unmount(self) -> None:
        bot._stop_event.set()


if __name__ == "__main__":
    LLMBotApp().run()
