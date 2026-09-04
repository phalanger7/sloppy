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
from textual.containers import Horizontal, Vertical
from textual.message import Message
from rich.text import Text
from rich.markup import escape
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog, Static

import llmbot_core as bot


class LogLine(Message):
    """A line for the log pane, posted from the background thread."""

    def __init__(
        self,
        text: str,
        *,
        action: bool = False,
        speak: bool = False,
        warning: bool = False,
    ) -> None:
        super().__init__()
        self.data = text
        self.is_action = action
        self.is_speak = speak
        self.is_warning = warning


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
    vsrc = snap["vision_source"]
    if vsrc == "auto":
        vision = f"auto ({'enabled' if snap['vision'] else 'disabled'})"
    else:
        vision = f"{vsrc} (forced)"
    # Below the Summary line, show the rolling summary (word-wrapped by the
    # Static) and the highlights, so the right panel carries the summarizer's
    # output, not just counts. Escaped: these are arbitrary LLM output and the
    # status Static renders markup.
    summary_text = snap["summary"].strip()
    summary_detail: list[str] = []
    if summary_text:
        age_min = round(snap["summary_age"] / 60)
        summary_detail = [
            f"Summary (made {age_min} min ago):",
            escape(summary_text),
            "",
        ]
        for h in snap.get("highlight_list", []):
            summary_detail.append(f"- {escape(h)}")
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
        f"Vision      : {vision}",
        (f"Summary   : {snap['highlights']} highlights, "
         f"{snap['pending_summary']} pending" if summary_text
         else f"Summary   : none yet ({snap['pending_summary']} pending)"),
        *summary_detail,
    ]
    return "\n".join(lines)


# Static hint row pinned to the bottom of the status pane (see CSS #status-hints).
_STATUS_HINTS = (
    "I = Inspect last LLM call\n"
    "V = Toggle vision\n"
    "P to pause\n"
    "Q = Quit"
)


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
    }
    #status-info {
        /* Flows naturally from the top of the status pane. */
    }
    #status-hints {
        /* Pinned to the bottom via dock (align is a no-op in this layout). */
        dock: bottom;
    }
    """

    BINDINGS = [
        ("i", "show_llm_debug", "Inspect LLM call"),
        ("I", "show_llm_debug", "Inspect LLM call"),
        ("v", "toggle_vision", "Toggle vision"),
        ("p", "toggle_pause", "Pause"),
        ("q", "quit", "Quit"),
        ("Q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield RichLog(id="log", auto_scroll=True, highlight=False)
            with Vertical(id="status"):
                yield Static(id="status-info")
                yield Static(id="status-hints")

    def on_mount(self) -> None:
        # Route the core's meaningful lines into the log via thread-safe
        # messages. The raw IRC sink is dropped — it is noise, not bot state.
        bot.irc_sink = lambda _text: None
        bot.action_sink = self._on_action
        bot.chat_sink = self._on_chat
        bot.speak_sink = self._on_speak
        bot.warning_sink = self._on_warning
        bot.debug_sink = lambda _text: None
        self._bot_thread = threading.Thread(target=bot.main, daemon=True)
        self._bot_thread.start()
        self.set_interval(1.0, self._refresh_status)
        self._refresh_status()

    def _on_action(self, text: str) -> None:
        self.post_message(LogLine(text, action=True))

    def _on_speak(self, text: str) -> None:
        self.post_message(LogLine(text, speak=True))

    def _on_chat(self, text: str) -> None:
        self.post_message(LogLine(text, action=False))

    def _on_warning(self, text: str) -> None:
        self.post_message(LogLine(text, warning=True))

    def on_log_line(self, msg: LogLine) -> None:
        log = self.query_one("#log", RichLog)
        if msg.is_speak:
            # The bot actually spoke: bright blue + bold, so it stands out
            # from the yellow action lines and the plain chat lines. "light_blue"
            # is not a recognised Rich style name (it rendered as plain white).
            log.write(Text(msg.data, style="bold bright_blue"))
        elif msg.is_warning:
            # A rejected summarizer output: bold red, so it stands out from the
            # yellow action lines and the blue speak lines.
            log.write(Text(msg.data, style="bold red"))
        elif msg.is_action:
            # Bold + bright so the bot's own actions stand out from chat.
            # A Text object (not a markup string) keeps brackets in lines
            # like "[AI] ..." literal instead of being parsed as tags.
            log.write(Text(msg.data, style="bold bright_yellow"))
        else:
            log.write(Text(msg.data))

    def _refresh_status(self) -> None:
        snap = bot.status_snapshot()
        self.query_one("#status-info", Static).update(_format_status(snap))
        self.query_one("#status-hints", Static).update(_STATUS_HINTS)

    def action_show_llm_debug(self) -> None:
        """Pop up the last LLM call for inspection (press 'i'/'I')."""
        self.push_screen(LLMDebugView())

    def action_toggle_vision(self) -> None:
        """Cycle the vision mode auto -> on -> off (press 'v').

        auto follows the server probe (does the loaded model see images?); on
        and off force the behaviour regardless of what the probe reports.
        """
        bot._cycle_vision_override()

    def action_toggle_pause(self) -> None:
        """Pause/unpause the bot (press 'P'). While paused it makes no LLM calls
        and no greetings; press 'P' again to resume."""
        bot._toggle_pause()

    def on_unmount(self) -> None:
        bot._stop_event.set()


class LLMDebugView(ModalScreen[None]):
    """Modal pop-up: everything passed to and returned by the last LLM call.

    Shows the system prompt, the messages exactly as sent to the model, the
    user prompt, and the returned output. Scroll with the wheel/arrows; close
    with Esc, the X key, or the close button. Literal text (the message repr
    contains brackets) is shown verbatim via markup=False.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    CSS = """
    #llm_debug {
        height: 85%;
        width: 90%;
        border: thick $error;
        border-title-background: $warning;
    }
    #close-btn {
        dock: bottom;
        margin: 1;
        width: 14;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="llm_debug", auto_scroll=True, markup=False, wrap=True)
        yield Button("Close  (Esc)", id="close-btn")

    def on_mount(self) -> None:
        self.border_title = "Last LLM call"
        self.query_one("#llm_debug", RichLog).write(bot.get_last_llm_call())

    def on_button_pressed(self) -> None:
        self.dismiss()


if __name__ == "__main__":
    LLMBotApp().run()
