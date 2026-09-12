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
- 2026-09-19: Easier to reach, and three new personas. Four things.

  (1) MENTIONS. Saying the nick mid-sentence is no longer ignored. Three tiers
  in `_resolve_prompt`: a leading/trailing nick stays a certain trigger; a
  mid-sentence mention from the person the bot was last talking to (within
  `mentions.certain_within_seconds`, 90s) is treated as addressed, rate limit
  bypassed; anything else replies at `mentions.reply_chance` (0.3). The
  unprompted guards still apply to the third tier, so the chance sets the
  flavour and not the volume. `_mention_is_certain` was first written as "the
  bot spoke in the last 90s", which made a mention certain for EVERYONE for 90
  seconds after any reply and swallowed the chance tier -- it is tied to the
  sender now, via a new `_conversation["at"]`.

  (2) DIRECTIVES are config. `[directives]` maps word -> mode and the regex is
  rebuilt from it, so the phrasings a channel actually uses are a file edit.
  Added `facts`, `factual` and `seriously`. One fix fell out: the subject-verb
  guard ("research shows ...") also ate "sloppy facts, are whales mammals",
  because "facts are" looks like a statement -- a directive word that leads the
  ask, with the bot addressed, now beats that guard.

  (3) SCHEDULED MOODS. Each `[moods.X]` may carry `minutes_per_hour`;
  `_plan_mood_window` lays them out at random non-overlapping moments, redrawn
  every window, so the timetable cannot be learned. Gaps are drawn rather than
  starts, which keeps the budgets exact by construction; verified 10/5/5 over
  six hours. A mood somebody asked for always wins -- the schedule only fills
  resting time. Two new personas, `mean` (crude, harder-swearing banter; the
  "drop it if somebody is genuinely having a bad time" rule outranks the rest)
  and `wholesome` (specific warmth, explicitly not saccharine), at 5 min/hour
  each, factcheck at 10.

  (4) `!quote` and `!buddha`. The first bang commands that are complete with no
  argument (`_BANG_DEFAULTS`); every other command still needs its subject.
  Both are in STRICT_MODES -- a misquote is a wrong answer, not a style -- and
  in a new CONTEXTLESS_MODES, because handing a recital the channel's last
  twenty lines had it ending a Buddhist teaching with "apply this to your four
  hours of renaming photos".

  Measured on the accuracy problem: the buddha prompt originally named the
  hot-coal line as a known fabrication, and the model produced exactly that
  fabrication anyway, with an invented citation. Removing the named example
  scored 2/35 against 5/35 for keeping it. Not significant at that n, but
  consistent across two runs, and the mechanism (a named example primes it) is
  the expected one, so the specific example is gone and the general warning
  stays. Misattribution is NOT solved: a 35B recalling quotations gets
  attributions wrong, and !quote produced a well-known Feynman misattribution
  in the first handful of samples. Treat both commands as entertainment.

  Also fixed: seven summarizer tests hardcoded six pending lines and broke the
  moment `memory.summary_min_lines` was raised to 10 in sloppy.toml; they
  derive the count now. And the mood schedule made a resting-mood test a coin
  flip, so `_no_scheduled_moods` pins it. 733 tests, green six runs running.
- 2026-09-18: Fixed a bug I introduced yesterday: the test suite was writing
  its fixtures into the LIVE state directory. `chatlog.jsonl` and
  `memory.json` were derived from `_profile_path` at import time, before
  `setUpModule` redirects it, so every `./check.sh` run appended to the real
  files -- 11451 lines of alice/bob/Probe0 in the channel log and a
  `{"summary": "NEW"}` rolling memory. With recall on, the bot started quoting
  "alice" into the channel, which is how it was caught. The suite already had
  this exact guard for the profile store, from the last time it happened.

  Both paths are now functions deriving from `_profile_path` when they are
  used, so redirecting that one path covers every store and the next one added
  is covered for free. `TestStateIsolation` asserts the derivation rather than
  a list of paths, and a gate run now provably leaves both files byte-identical.
  The polluted files were checked line by line (0 records not attributable to a
  fixture) and moved aside as `*.test-polluted` rather than deleted.

  Also added the TUI toggle: `l`/`L` cycles recall config -> on -> off,
  following the vision override's three states so a runtime toggle and a config
  reload cannot disagree about which is in charge. The status row now names
  both the state and where it came from -- `on (config)`, `on (forced)`,
  `off (config, still logging)` -- because the previous row signalled "on" by
  the absence of a note, which is indistinguishable from a row you have not
  understood. 700 tests, gate green.
- 2026-09-17: Long-term recall, behind `[recall] enabled` and OFF by default.
  New `recall.py` (its own module for the same reason `profiles.py` is one: a
  store with its own persistence and ~200 lines of scoring). Every non-trivial
  channel line is appended to `chatlog.jsonl` beside the profiles, one JSON
  record per line, each carrying `v` so a schema change is skippable rather
  than fatal. At reply time `_recall_section` scores the log against the
  current prompt plus the last `query_lines` channel lines and injects up to
  `passages` hits, each with the line either side, as
  `--- EARLIER IN THE CHANNEL --- ` ahead of the recent chat so the block reads
  oldest to newest.

  Scoring is Okapi BM25 with three things layered on, and the layers matter
  more than the formula. (1) Query terms appearing in more than
  `COMMON_TERM_RATIO` (8%) of the log are dropped: at channel scale IDF alone
  leaves "the" and "out" enough weight to outscore the one rare word the
  question was about -- measured, this was the difference between 5/5 topics
  retrieved and 3/5 with 2 wrong. (2) The score is divided by the best a single
  line could score for that query, so the floor is a fraction and does not need
  re-tuning as the log grows; a raw BM25 threshold drifts with IDF. (3) A
  recency half-life. The half-life started at one day, which put everything
  older than the recent-line buffer out of reach -- i.e. defeated the whole
  feature -- and is now 14 days.

  Calibrated on a 1500-line synthetic channel with five known topics and
  Zipf-shaped filler: every topic retrieved correctly and none wrongly for any
  floor between 0.1 and 0.4, recall falling off above 0.5, so the default is
  0.3. Cost on the reply path is 34 ms at the 20000-line cap.

  Capture is deliberately NOT gated by the flag -- switching recall on against
  an empty log would mean waiting a fortnight to learn whether it was any good.
  `!forget` now erases that person from the log and rewrites it, or the bot
  could quote somebody next Tuesday that it promised to forget today. The
  exclusion of already-shown lines is by timestamp, not by count: the log spans
  restarts and the prompt's recent block does not, so they are not the same
  tail of the same list. Verified live -- with recall on the bot answered a
  photo-renaming line with the exiftool command from six days earlier; with it
  off, it could not. TUI status gains a Recall row. 689 tests, gate green.

  Also fixed a real flake found on the way: 20 TUI tests waited on a fixed
  `asyncio.sleep` after `simulate_key`. One failed in a gate run and then
  passed alone, in collection order, and across six shuffled orderings. They
  now use `_settle(ctx)`, which drains Textual's message queue instead of
  watching the clock. The original failure was never reproduced -- including
  under 24x CPU load, where both versions passed 6/6 -- so the sleep is the
  suspect, not the proven cause; what changed is that the clock is no longer
  part of the answer.
- 2026-09-16: Channel memory now survives a restart, and the prompt knows what
  time it is. `_rolling` gained an `at` (wall clock) and is written to
  `<XDG_DATA_HOME>/sloppy/memory.json` -- version 1, beside the profiles, via
  the profile store's atomic writer -- when a summary lands and again on
  shutdown, behind a `_memory_dirty` flag so `shutdown()` stays idempotent.
  `_load_memory` runs beside `_load_profiles` at startup; a missing, corrupt or
  wrong-version file starts fresh, and every field is type-checked on the way
  in. Before this, a restart wiped everything older than the recent-line
  buffer, which is most of what the bot knew. The context block gained a
  `--- NOW ---` section (time and date), a `[HH:MM]` stamp on each recent line
  from a new `_recent_times` deque, and an age on the memory header
  (`CONVERSATION MEMORY (last updated 2 hours ago)`) so a restored summary is
  not read as current. `status_snapshot`'s `summary_age` now comes off the wall
  clock rather than the monotonic trigger clock, which would have called a
  restored summary "0s old". The summarizer's input is unchanged: its prompt
  describes lines as "nick: what they said" and that prompt is hand-tuned.
  Highlights carry no per-item time -- the summarizer rewrites the list whole
  each tick, so there is no stable identity to stamp. 663 tests, gate green.
  `bot.py` untouched.
- 2026-09-15: Closed the hole the previous entry left: both drafts could be
  rejected and the room then heard a brain-offline line instead of a reply
  (Alexander caught one in the log pane). Two causes. (1) A redraw was the
  identical request, so a model that had fallen into writing transcript had
  nothing pushing it back out. `_with_retry_nudge` now appends a correction to
  the user turn -- keyed to the reason, `transcript` or `echo` -- rather than a
  second system message, since the templates that matter refuse a system
  message that is not first. Measured on the scenario from the log, pooled over
  two runs: plain 8/100 drafts rejected, nudged 3/100. A third arm that dropped
  the recent-chat section on redraw did as well (0/60) but loses the grounding,
  so it was not taken. (2) `_looks_like_transcript` counted "nick:" only at the
  start of a LINE, and the dumps arrive as one unbroken line -- every one of
  them was being caught by `_echoes_recent` instead, which worked but logged
  the wrong reason and sent the wrong nudge. It now counts nick-colons anywhere
  in the text (`_NICK_ANYWHERE_RE`); the threshold of 2 still lets a single
  address through for `_strip_nick_prefix` to tidy. End to end after: 50
  interjections on the same scenario, 1 draft rejected, 0 give-ups (before, 20
  interjections gave up once). `personality.attempts` left at 2. 650 tests,
  gate green. `bot.py` untouched.
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


