# irc_llm_bot — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 200 tests pass, quality gate clean. 203 tests pass. Two answering modes: chat (channel persona) and factual (`factcheck`, `science:`, `research:` — verdict word on claims, plain answer on questions, no jokes). A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (25s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. After `SILENCE_TIMEOUT` (30 min) with nobody talking it breaks the silence and opens the floor for `OPEN_FLOOR_WINDOW` (60s), answering anything from anyone up to `OPEN_FLOOR_MAX_PROMPTS` (8). After `IDLE_INTERJECT_AFTER` (20) unaddressed channel lines the bot chimes in unprompted, 50/50 between reacting to the last line and being asked for `IDLE_PROMPT`, under a `MODE_INTERJECT` prompt that leans to banter, tells a joke now and then, and deliberately leaves room for random tangents. Auto-interject openers wait `JOIN_GRACE_PERIOD` (10s) after JOIN before firing, on both the silence-breaker and idle-interject paths, so the userlist has arrived first (the bot used to invent names on join). LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise. When it does mention someone, the user list woven into the prompt is ordered by relevance, not registration order: the person who addressed the bot or spoke most recently first (~70% of mentions), then recent speakers from the last 200 lines (~20%), then the rest of the channel (~10%), with a "prefer the first name" instruction steering the persona. On join, before any line has been spoken, that recent tier is empty so the slots fall through to other members (a random name), exactly as intended. The last 200 channel lines are also fed into the LLM call as real chat history rather than pasted into the prompt: `_recent_messages()` returns them as `user` messages whose content is `"<sender>: <text>"` inline (oldest first — the sender is written into the content, not a separate field, so it's portable and open models parse it well) and `_call_llm` inserts them between the system prompt and the user's message on every mode (factual included), logging `Injected XX lines of chat history as context` to the terminal at call time. On top of the per-reply modes there are three global moods: `banter`, `serious` and `factcheck`, switched by the bare word (`serious`, `Heretic: factcheck`, `AI: banter`) and announced in-channel without an LLM call ("Ok I'll be serious for a while", "Oh you want bants huh? Fine", "Factchecking engaged"). Serious and factchecking swap the chat and interjection personas for `MODE_SERIOUS` / `MODE_FACTUAL` (`MOOD_MODES`) until someone names another mood or `MOOD_TIMEOUT` (15 min) passes; banter is the resting state and never expires. `factcheck <claim>` is still the one-off it always was, and any message that names a mode itself ignores the mood. The boot mood is a coin flip between banter and serious (`_random_mood`) — never factchecking. Addressed to the bot, a mood command may carry filler ("Heretic, be serious for once"), unaddressed only the bare word counts. Addressing also tolerates a greeting before the nick ("hey Heretic.. whats up") and any of `:,;.!?-` after it.

TUI: `llmbot_tui.py` is a Textual front-end importing `llmbot_core` (a fork of bot.py) that paints two panes — top-left = a scrollable log of everything worth seeing, top-right = status (mood/mode, chat-history buffer count, open-floor, chatter count, join/reply/quiet timers). The log shows the bot's own actions (being addressed, switching mood, interjecting, injecting chat history, shutdown) in **bold bright-yellow**; the line the bot actually speaks is rendered **bright blue + bold** instead (a dedicated `speak()` sink in the core, routed through in `_process_pending`) so it is easy to tell a line of banter apart from a bot status line; and every line of channel chat that enters the history buffer as plain `nick: text`. The status pane is a `Vertical`: the dynamic indicators flow from the top and a static hint row ("press D to inspect last LLM call" / "press Q to quit") is pinned to the bottom via `dock: bottom` (`align` is a no-op in this stream-layout Textual 8.2.8, and `grow`/`dock` had to be verified against the installed version). The speak style was fixed from `light_blue` (not a valid Rich style name — it rendered as plain white) to `bold bright_blue`, and the debug modal's `RichLog` uses `wrap=True` so long lines wrap instead of running off. The raw IRC log is intentionally not shown (noisy, doesn't affect the bot); the core's `irc_sink` is a no-op. A new `chat()` sink posts the `nick: text` line from `_note_recent` so the log mirrors the history exactly. The bot runs in a daemon thread; sinks post thread-safe `LogLine` messages back to the UI thread (action vs chat flagged so the handler can style them), and status refreshes every 1.0s from `status_snapshot()`. Run with `python3 llmbot_tui.py` (Textual is in system Python 3.14.7, no venv). bot.py is untouched — all TUI logic lives in `llmbot_core.py` + `llmbot_tui.py`. The status pane's `Users` chatter list is ordered by recency (most recently spoken/engaged first) via `_mention_targets_locked()`, matching the order the names are handed to the LLM; members who have not spoken fall to the end in registration order. The chatter list already excluded the bot's own nick; the 3rd-person self-talk came from the LLM history buffer re-feeding the bot's own echoed messages, fixed 2026-09-01 (see Recent history).

Image analysis is on demand. A vision model is auto-detected by probing the server's `/props` endpoint for `modalities.vision` (run at boot and every 2s in the poll loop); the TUI `v` key cycles it auto -> on -> off (a manual override that wins over the probe), shown in the status pane as `auto (enabled/disabled)` or `on/off (forced)`. Requests are either a command (`!image <url>`, `!img`, `image:`) or a referential "what's in the image Tim just posted" (resolved from a per-nick `_recent_images` index, falling back to the channel's latest). The URL rides on the user message as an `image_url` content part to the shared :8080 server/model (one llama-server, the same Tiel-Coder model with `--mmproj`), answered in a new `MODE_VISION` persona — banter by default, accurate/naming when the image needs it — kept out of `MOOD_MODES` so global moods don't override it. The command trigger is loud (`!image`/`!img`) or needs a URL so ordinary chat is not matched; a referential request that cannot resolve a URL falls through to ordinary handling. 240 tests, gate green. `bot.py` untouched.

Newcomers and returnees are now greeted. On JOIN the bot welcomes a nick in-channel (50% of the time with a mild roast), skipping the greeting when the nick left only a few chatlines ago (tracked via QUIT/PART + a chatline counter, so a frequent pop-in is not greeted each time). Anyone who speaks up after IDLE_GREET_AFTER (2h) of silence gets a "back again" welcome (again 50% roast). Greetings are templated, sent as channel PRIVMSGs from the receiver, and unconditional of the current mood. Last-seen per nick is recorded in `_note_recent` and reset on JOIN so a rejoin is not also read as idle. `LLM_MODEL` is now `qwen35-9b` (the llama.cpp `-alias`), correcting the old misleading `llama-3.2-3b-instruct`. 266 tests, gate green. `bot.py` untouched.

Pressing `P` in the TUI pauses the bot: while paused it makes no LLM calls and
no greetings (join, idle-return, and interjections all stay silent) until `P`
is pressed again. The gate lives in `_process_pending`/`_process_pending_vision`
(each returns early when `_paused["on"]` is set by `_toggle_pause`), so any
queued request waits for unpause; the single-slot pending buffer means only the
latest request fires when resumed. The TUI `p` key calls `_toggle_pause` and the
status hint row now shows "P to pause". Also: the LLM chat-history buffer grew
from 100 to 200 lines (`RECENT_LINES`), and one-word lines shorter than
`MIN_CHAT_CHARS` (10) are no longer stored in that buffer (via `_is_trivial_message`
in `_note_recent`) — they still count toward chatline/idle timing and are still
logged to the pane, just not fed to the model. 281 tests, gate green. `bot.py`
touched.

## Known issues / open questions
- Uses raw TCP (not `irc` lib) due to Python 3.14 incompatibility with `tempora` dependency.
- No PING/PONG handling yet — may time out on long idle. (The receiver does answer
  PING; this note is about idle timeouts, not parsing.)
- `LLM_MODEL` is still `"llama-3.2-3b-instruct"` while the server actually serves
  `qwen35-9b`. llama.cpp ignores the field, so this is cosmetic, but it is
  misleading and worth correcting.
- `IRC_MAX_LEN = 400` assumes a worst-case ~100-byte server hostmask prefix. It
  has not been measured against hive.2bd.net's actual prefix length.
- `LLM_TEMPERATURE` is pinned (currently 1.2) per request rather than inheriting  
  the server's `--temp`, so the persona does not drift when the server is retuned  
  for other models. Intended long-term direction (Alexander, 2026-08-28) is to  
  eventually drop the parameter and inherit instead. The guard test only checks  
  the pin is a valid sampling value (0–2), not a specific number, so the pin can  
  change without a test rewrite.

## Recent history (last 5 entries, oldest dropped)
- 2026-09-06: Added a `P` pause key and trimmed LLM context. Pressing `P` in
  the TUI sets `_paused["on"]` (via `_toggle_pause`); `_process_pending` and
  `_process_pending_vision` return early while paused, so the bot makes no LLM
  calls and no greetings (join/idle/interject all stay silent) until `P` is
  pressed again. Queued requests wait in the single-slot pending buffer, so
  only the latest fires on resume. The TUI `p` binding calls `_toggle_pause`
  and the status hint row gained "P to pause". `RECENT_LINES` raised 100 ->
  200, and `_is_trivial_message` (via `_note_recent`) stops storing one-word
  or <10-char lines in the chat-history buffer — they still count toward
  chatline/idle timing and are still logged, just not fed to the model.
  Added TestTrivialMessageFilter, TestPause, TestPauseTUI; updated
  TestCoreSelfFiltering for non-trivial words. 281 tests, gate green.
  `bot.py` untouched.
- 2026-09-05: Greet newcomers and returnees. On JOIN the bot welcomes a nick
  in-channel (50% with a mild roast), skipping when the nick left only a few
  chatlines ago (QUIT/PART recorded via `_handle_quit` + a `_chatlines` counter,
  compared in `_join_greeting_text`). Anyone who speaks up after `IDLE_GREET_AFTER`
  (2h) of silence gets a "back again" welcome (`_note_recent` tracks per-nick
  `_last_seen`, reset on JOIN so a rejoin is not also read as idle). Greetings
  are templated channel PRIVMSGs, unconditional of mood, built by `_greeting_text`
  (greeting pool + roast pool, roast when `random() < GREET_ROAST_CHANCE`). The
  receiver dispatches JOIN/QUIT/PART via a new `_split_event` + `_handle_line`
  helper (extracted to keep `receiver` under the complexity ceiling). `LLM_MODEL`
  set to `qwen35-9b` (the llama.cpp `-alias`), replacing the misleading
  `llama-3.2-3b-instruct`. Added TestSplitEvent, TestGreetingText, TestJoinGreet,
  TestQuitTracking, TestIdleGreet, TestReceiverGreetIntegration. 266 tests, gate
  green. `bot.py` untouched.
- 2026-09-04: On-demand image analysis via the vision model. Auto-detects a
  loaded vision model by probing the server `/props` (`modalities.vision`) at
  boot and every 2s; the TUI `v` key cycles auto -> on -> off (forced override),
  shown in the status pane. Triggers: command (`!image`/`!img`/`image:` + URL)
  and referential ("what's in the image Tim just posted", resolved from a
  per-nick recent-image index with a channel-latest fallback). Image rides on
  the user message as an image_url content part to the shared :8080 server/model
  (same Tiel-Coder model with `--mmproj`), answered in a new MODE_VISION persona
  (banter by default, accurate when naming) kept out of MOOD_MODES. Added
  `_extract_image_urls`, `_match_vision_trigger`, `_call_llm_vision`,
  `_probe_vision`, vision override/cycle + status fields. 240 tests, gate
  green. `bot.py` untouched.
- 2026-09-04: Status-pane hint row shortened to "D = Inspect last LLM call" /
  "Q = Quit". Inspection hotkey moved D -> I (i/I), and the LLM debug modal
  dropped its X key (Escape + close button only); close button text is now
  "Close  (Esc)". Updated TestTUIStatusNote, TestTUIStyleFixes (i key), and
  TestLLMDebugModal (i/I open, escape/button close, X test removed). 203 tests,
  gate green. `bot.py` untouched.
- 2026-09-03: TUI status pane restructured so the hint row is pinned to the
  bottom. The status pane is now a `Vertical` with a flowing `#status-info`
  Static (indicators, top) and a `dock: bottom` `#status-hints` Static. The
  hints moved out of `_format_status()` into a `_STATUS_HINTS` constant
  (`align` is a no-op in this stream-layout Textual 8.2.8; verified that only
  `grid` layouts honour `align`/`dock`, and `dock: bottom` works via
  `_arrange._arrange_dock_widgets`). Speak style fixed `light_blue` -> `bold
  bright_blue` (the former is not a valid Rich style name and rendered plain
  white); debug modal `RichLog` gained `wrap=True`. Added `TestTUIStyleFixes`
  (3 tests: speak is blue+bold, hints at pane bottom / indicators at top,
  modal wraps). 203 tests, gate green. `bot.py` untouched.

