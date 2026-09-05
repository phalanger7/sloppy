# sloppy — session notes

## Current status
New standalone `summarizer.py`: rolling IRC summarizer, sole public fn `summarize_tick(prev_summary, prev_highlights, new_lines) -> (summary, highlights)` via OpenAI-compatible `:8080` JSON-schema response. Never raises (returns inputs on any failure); empty `new_lines` short-circuits with no server call. Now integrated into `llmbot_core` (see Recent history 2026-09-10): a second plain-list buffer `_pending_summary_lines` records every IRC line alongside the existing 200-line chatter buffer; a daemon worker `_summarize_loop` runs every `SUMMARIZE_INTERVAL` (600s), snapshots + clears pending under `_prompt_lock`, summarizes the snapshot outside the lock, and replaces rolling `_rolling{"summary","highlights"}`. The normal chat prompt now folds summary + highlights + the last 20 IRC lines into one system (background/observation) message via `_context_block` (recent chat kept as `sender: text`, never an LLM role), with the current event as the sole user message. The TUI status pane shows the rolling summary/highlights + pending count. 303 tests, gate green. `bot.py` untouched.

Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 200 tests pass, quality gate clean. 203 tests pass. Two answering modes: chat (channel persona) and factual (`factcheck` — verdict word on claims, plain answer on questions, no jokes). `science` / `research` / `answer` are directive modes that answer seriously and concisely WITHOUT a TRUE/FALSE verdict (unlike factcheck); recognised with leniency ('Research dangers of lead', 'sloppy can you answer this') by a dedicated `_match_directive`, which refuses when the command word is a noun ('the answer to...') or the subject of a statement ('research shows...'); all three share one context-free persona. A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (25s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. After `SILENCE_TIMEOUT` (30 min) with nobody talking it breaks the silence and opens the floor for `OPEN_FLOOR_WINDOW` (60s), answering anything from anyone up to `OPEN_FLOOR_MAX_PROMPTS` (8). After `IDLE_INTERJECT_AFTER` (20) unaddressed channel lines the bot chimes in unprompted, 50/50 between reacting to the last line and being asked for `IDLE_PROMPT`, under a `MODE_INTERJECT` prompt that leans to banter, tells a joke now and then, and deliberately leaves room for random tangents. Auto-interject openers wait `JOIN_GRACE_PERIOD` (10s) after JOIN before firing, on both the silence-breaker and idle-interject paths, so the userlist has arrived first (the bot used to invent names on join). LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise. When it does mention someone, the user list woven into the prompt is ordered by relevance, not registration order: the person who addressed the bot or spoke most recently first (~70% of mentions), then recent speakers from the last 200 lines (~20%), then the rest of the channel (~10%), with a "prefer the first name" instruction steering the persona. On join, before any line has been spoken, that recent tier is empty so the slots fall through to other members (a random name), exactly as intended. The last 200 channel lines are also fed into the LLM call as real chat history rather than pasted into the prompt: `_recent_messages()` returns them as `user` messages whose content is `"<sender>: <text>"` inline (oldest first — the sender is written into the content, not a separate field, so it's portable and open models parse it well) and `_call_llm` inserts them between the system prompt and the user's message on every mode (factual included), logging `Injected XX lines of chat history as context` to the terminal at call time. On top of the per-reply modes there are three global moods: `banter`, `serious` and `factcheck`, switched by the bare word (`serious`, `Heretic: factcheck`, `AI: banter`) and announced in-channel without an LLM call ("Ok I'll be serious for a while", "Oh you want bants huh? Fine", "Factchecking engaged"). Serious and factchecking swap the chat and interjection personas for `MODE_SERIOUS` / `MODE_FACTUAL` (`MOOD_MODES`) until someone names another mood or `MOOD_TIMEOUT` (15 min) passes; banter is the resting state and never expires. `factcheck <claim>` is still the one-off it always was, and any message that names a mode itself ignores the mood. The boot mood is a coin flip between banter and serious (`_random_mood`) — never factchecking. Addressed to the bot, a mood command may carry filler ("Heretic, be serious for once"), unaddressed only the bare word counts. Addressing also tolerates a greeting before the nick ("hey Heretic.. whats up") and any of `:,;.!?-` after it.

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
untouched.

Most recently the summarizer trigger became a hybrid: it fires when more than
600s OR more than 25 lines have passed since the last summary, but only if at
least 5 lines have accumulated since it (`SUMMARIZE_VOLUME_LINES=25`,
`SUMMARIZE_MIN_LINES=5`; `SUMMARIZE_INTERVAL=600` kept as the age arm), and the
daemon loop now polls every `SUMMARIZE_POLL_INTERVAL=15`s instead of sleeping
600s. The TUI status pane shows the summarizer's actual content below the count
line — `Summary (made X min ago):`, the summary word-wrapped on the next line, a
blank line, then each highlight as `- ` — fed from a new `highlight_list` field
in `status_snapshot`, rendered via `rich.markup.escape` since it is arbitrary
LLM output. The bot now boots into banter mode rather than a coin flip between
banter and serious (`_random_mood`). The summarizer's `SYSTEM_PROMPT` was
replaced with a structured rolling-memory prompt: a 400-700 char rolling summary
plus up to 5 highlights, with explicit guidance to carry forward useful context,
drop stale items, and never invent facts; rendered as a triple-quoted string.
The `summarize_tick` contract is unchanged. The model's summary is validated
before it is stored: if it is not a string, is empty (whitespace-only), or
exceeds 1200 characters it is rejected and the previous one kept, and a red
`INVALID SUMMARY RECEIVED: <reason>` line is posted to the log pane via a new
`warning` sink (`bold red`). 316 tests, gate green. `bot.py` untouched.

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
- 2026-09-13: Added validation of the model's summary before it overwrites the
  rolling one. `_reject_reason` rejects a summary that is not a string, is
  empty (whitespace-only), or exceeds `SUMMARIZE_MAX_CHARS` (1200); on rejection
  the previous summary is kept, the new one discarded, and `warning()` posts a
  red `INVALID SUMMARY RECEIVED: <reason>` line to the TUI log pane. New
  `warning_sink`/`warning()` core sink plus a `warning` branch on `LogLine`
  (bold red). 316 tests, gate green.
- 2026-09-12: Replaced the summarizer's `SYSTEM_PROMPT`. The old one was a single
  line ("You are an IRC log summarizer. Output ONLY valid JSON..."). The new one
  is a structured rolling-memory prompt: a ~400-700 char rolling summary + up to
  5 highlights, with explicit guidance to carry forward useful context, replace
  stale items, and never invent facts (JSON output schema spelled out). Rendered
  as a triple-quoted string; `summarize_tick` contract unchanged. 306 tests,
  gate green.
- 2026-09-11: Summarizer trigger became a hybrid: fire when >600s OR >25 lines
  since the last summary, but only if >=5 lines have accumulated since it
  (`SUMMARIZE_VOLUME_LINES=25`, `SUMMARIZE_MIN_LINES=5`; `SUMMARIZE_INTERVAL=600`
  kept as the age arm). The daemon loop polls every `SUMMARIZE_POLL_INTERVAL=15`
  s instead of sleeping 600s. TUI status pane gains the summarizer's real content
  below the count line: `Summary (made X min ago):`, the summary word-wrapped on
  the next line, a blank line, then each highlight as `- ` (built from a new
  `highlight_list` field in `status_snapshot`, escaped via `rich.markup.escape`
  since it is arbitrary LLM output). The bot now boots into banter mode (was a
  coin flip between banter/serious in `_random_mood`). TestSummarizerIntegration
  updated: age-trigger test, pause now overrides a valid trigger, and new
  volume-trigger, min-lines-gate, and mid-range-no-op tests. 306 tests, gate
  green. `bot.py` untouched.
- 2026-09-10: Integrated the rolling summarizer into `llmbot_core` and the TUI
  status pane. Every IRC line now goes into BOTH the existing 200-line chatter
  buffer AND a second plain-list buffer `_pending_summary_lines` (no deque; the
  shared `_prompt_lock` only). A daemon worker `_summarize_loop` runs every
  `SUMMARIZE_INTERVAL` (600s), gated by `_paused["on"]`; `_summarize_pending`
  snapshots + clears pending under the lock, calls `summarizer.summarize_tick`
  on the snapshot OUTSIDE the lock (so IRC keeps appending during generation),
  then stores the new rolling `_rolling{"summary","highlights"}` (kept in a
  container to avoid a global reassignment, ruff PLW0603) + `_last_summary_at`.
  The normal `_call_llm` now folds summary + highlights + a verbatim sample of
  the last 20 IRC lines into ONE system (background/observation) message via
  `_context_block`, keeping the current event as the sole user message -- the
  summarizer output and recent chat are presented as room context, not as
  separate user messages. Each IRC line keeps its `sender: text` form (sender
  inline as the speaker's name, never an LLM role), so `alice: hi` not
  `user: alice: hi`; sections are dropped when empty. `_summary_block` removed
  and replaced by `_context_block`; `_recent_messages` unchanged (still used by
  `_call_llm_vision`). The TUI `status_snapshot` gains a Summary line (rolling
  summary + highlights + pending count + time since last summary). Added
  TestSummarizerIntegration (10 tests): fed-to-both-buffers, pending skips own
  nick + trivial, worker summarizes/clears, no-op when empty, skipped while
  paused, snapshot decoupled from the live list, single-system-message prompt
  layout, last-20 recent cap, and summarize_tick returning inputs on server
  error. 303 tests, gate green. `bot.py` untouched.
- 2026-09-09: Added `summarizer.py` (stdlib + `requests` only). `summarize_tick` POSTs prev rolling state + new lines to llama `:8080` with a json_schema `response_format`, returns updated `(summary, highlights)`. Defensive parse: json.loads in try/except, guards non-dict/missing keys/wrong types, strips+de-dups+cap-5 highlights, empty `new_lines` returns inputs without calling. Whole request+parsing in one `except Exception` -> stderr + inputs. `if __name__` demo for standalone smoke test. Not yet integrated into core; integration pending design alignment.
- 2026-09-07: Directive modes science/research/answer. These answer seriously
  and concisely like the fact-checker but WITHOUT a TRUE/FALSE verdict; all
  three share one context-free persona (`_serious_answer_prompt`), wired into
  `_system_prompt` and treated context-free in `_system_context`. New
  `_match_directive` recognises the command word spelled correctly with filler
  tolerated around it ('Research dangers of lead', 'hey sloppy, research this',
  'sloppy can you answer this or that'); it refuses when the word is a noun
  (preceded by an article) or the subject of a statement (followed by a verb,
  so 'research shows ...' stays ordinary chat). Wired into `_match_trigger`
  first; `factcheck` stays in `FACTUAL_TRIGGERS` and keeps its old behaviour in
  the frozen bot.py. `science experiments are controlled` is a known false
  positive (structurally identical to the wanted 'science whales are fish', so
  left firing rather than break the good case). Added TestDirectiveModes (12
  tests) against llmbot_core; existing TestFactualMode (bot.) untouched. Split
  `_match_trigger`'s trailing-nick handling into `_match_trailing_nick` to stay
  under the complexity ceiling. 293 tests, gate green. `bot.py` untouched.
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


