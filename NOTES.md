# irc_llm_bot — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 167 tests pass, quality gate clean. Two answering modes: chat (channel persona) and factual (`factcheck`, `science:`, `research:` — verdict word on claims, plain answer on questions, no jokes). A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (25s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. After `SILENCE_TIMEOUT` (30 min) with nobody talking it breaks the silence and opens the floor for `OPEN_FLOOR_WINDOW` (60s), answering anything from anyone up to `OPEN_FLOOR_MAX_PROMPTS` (8). After `IDLE_INTERJECT_AFTER` (20) unaddressed channel lines the bot chimes in unprompted, 50/50 between reacting to the last line and being asked for `IDLE_PROMPT`, under a `MODE_INTERJECT` prompt that leans to banter, tells a joke now and then, and deliberately leaves room for random tangents. Auto-interject openers wait `JOIN_GRACE_PERIOD` (10s) after JOIN before firing, on both the silence-breaker and idle-interject paths, so the userlist has arrived first (the bot used to invent names on join). LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise. When it does mention someone, the user list woven into the prompt is ordered by relevance, not registration order: the person who addressed the bot or spoke most recently first (~70% of mentions), then recent speakers from the last 100 lines (~20%), then the rest of the channel (~10%), with a "prefer the first name" instruction steering the persona. On join, before any line has been spoken, that recent tier is empty so the slots fall through to other members (a random name), exactly as intended. The last 100 channel lines are also fed into the LLM call as real chat history rather than pasted into the prompt: `_recent_messages()` returns them as `user` messages whose content is `"<sender>: <text>"` inline (oldest first — the sender is written into the content, not a separate field, so it's portable and open models parse it well) and `_call_llm` inserts them between the system prompt and the user's message on every mode (factual included), logging `Injected XX lines of chat history as context` to the terminal at call time. On top of the per-reply modes there are three global moods: `banter`, `serious` and `factcheck`, switched by the bare word (`serious`, `Heretic: factcheck`, `AI: banter`) and announced in-channel without an LLM call ("Ok I'll be serious for a while", "Oh you want bants huh? Fine", "Factchecking engaged"). Serious and factchecking swap the chat and interjection personas for `MODE_SERIOUS` / `MODE_FACTUAL` (`MOOD_MODES`) until someone names another mood or `MOOD_TIMEOUT` (15 min) passes; banter is the resting state and never expires. `factcheck <claim>` is still the one-off it always was, and any message that names a mode itself ignores the mood. The boot mood is a coin flip between banter and serious (`_random_mood`) — never factchecking. Addressed to the bot, a mood command may carry filler ("Heretic, be serious for once"), unaddressed only the bare word counts. Addressing also tolerates a greeting before the nick ("hey Heretic.. whats up") and any of `:,;.!?-` after it.

TUI: `llmbot_tui.py` is a Textual front-end importing `llmbot_core` (a fork of bot.py) that paints two panes — top-left = a scrollable log of everything worth seeing, top-right = status (mood/mode, chat-history buffer count, open-floor, chatter count, join/reply/quiet timers). The log shows the bot's own actions (being addressed, replying/speaking, switching mood, interjecting, injecting chat history, shutdown) in **bold bright-yellow**, and every line of channel chat that enters the history buffer as plain `nick: text`. The raw IRC log is intentionally not shown (noisy, doesn't affect the bot); the core's `irc_sink` is a no-op. A new `chat()` sink posts the `nick: text` line from `_note_recent` so the log mirrors the history exactly. The bot runs in a daemon thread; sinks post thread-safe `LogLine` messages back to the UI thread (action vs chat flagged so the handler can style them), and status refreshes every 1.0s from `status_snapshot()`. Run with `python3 llmbot_tui.py` (Textual is in system Python 3.14.7, no venv). bot.py is untouched — all TUI logic lives in `llmbot_core.py` + `llmbot_tui.py`.

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
- 2026-08-30: TUI left pane became a full log. Actions (the bot speaking,
  being addressed, switching mood, interjecting, injecting history, shutting
  down) render bold + bright-yellow; every line of chat that enters the history
  buffer also prints as plain `nick: text`. Added a `chat()` sink in the core
  (posts from `_note_recent`, so the log mirrors the history exactly) and a
  `chat_sink`; the `irc_sink` stays a no-op. The log pane is a Textual `RichLog`
  and lines are written as `Text` objects (not markup strings) so brackets in
  lines like `[AI] ...` stay literal. `on_log_line` is added to the vulture
  ignore list in pyproject.toml. 180 tests, gate green.
- 2026-08-30: Mentions stopped being random. `_mention_targets()` orders the
  channel user list by relevance -- the person who addressed the bot or spoke
  last first (~70%), then recent speakers from the last 15 lines, most recent
  first (~20%), then the rest (~10%) -- and `_system_context` names them in that
  order with a "prefer the first one" instruction, so the persona favours who
  engaged it instead of picking a name at random. The receiver now records each
  line's sender in a parallel `_recent_senders` deque (maxlen=15), threaded
  through `_note_recent(message, sender)`; when no lines have been spoken yet a
  fresh join, that tier is empty and those slots fall through to other members.
  `test_serious_mood_reaches_the_llm_call` now asserts against
  `_system_context(MODE_SERIOUS)` (serious still gets the userlist) rather than
  the context-free prompt. 178 tests.
- 2026-08-30: Chat history now goes into the LLM call as messages, not the
  system prompt. `_recent_messages()` returns the last 100 channel lines as
  `user` messages (oldest first) and `_call_llm` slots them between the system
  prompt and the user's message on *every* mode -- factual included -- because a
  reply is always inside an ongoing room; it prints
  "Injected XX lines of chat history as context" when it does. `_system_context`
  no longer appends "Recent channel messages:" (that block was removed). The
  buffers grew from 15 to 100 lines (`RECENT_LINES`). Recent-history tests moved
  from `_system_context` to asserting on the messages passed to `create`.
  179 tests.
- 2026-08-30: Textual TUI front-end. `llmbot_tui.py` imports `llmbot_core`
  (a fork of bot.py) and shows only bot-affecting actions + a status pane; the
  raw IRC log is dropped (its `irc_sink` is a no-op). The bot runs in a daemon
  thread, sinks post `ActionLine` messages to the UI thread, and status
  refreshes every second from `status_snapshot()`. New files: `llmbot_core.py`,
  `llmbot_tui.py`, `pyproject.toml` (vulture `ignore_names` for Textual's
  reflection attrs). `bot.py` unchanged. 180 tests.
- 2026-08-30: Recent-history senders are now encoded inline in each line's
  content ("alice: hi") rather than in a separate `name` field. `name` is
  OpenAI-specific and less portable across llama.cpp-style backends, and most
  open models are trained on inline-labeled chat data so they parse
  "alice: hi" more reliably. `_recent_messages()` builds `{"role": "user",
  "content": sender + ": " + text}` (sender omitted if unknown), and
  `test_recent_lines_have_no_name_field` locks in the absence of the field.
  180 tests.
- 2026-08-30: Fixed the join grace gate. `main()` assigned the join time to a
  *local* (`_joined_at = time.monotonic()`, no `global`), so the global stayed
  0.0 and `_within_join_grace()` was always False -- the opener fired before
  the 353 userlist arrived and invented usernames again. Fixed by recording the
  join time in a `_joined = {"at": ...}` container (matches the `_activity`
  idiom, avoids ruff PLW0603). Added `TestMainJoinGrace` which runs the real
  `main()` with a mocked socket and asserts the global is set (fails without the
  fix). 169 tests.
- 2026-08-30: The auto-interject grace gate now covers BOTH triggers. The
  premature join opener came from `_note_chatter` (the 20-line idle-interject
  path), which was not gated, so the bot still invented names on join. Both
  `_check_silence` and `_note_chatter` now early-return while
  `time.monotonic() - _joined_at < JOIN_GRACE_PERIOD` (bumped 7s -> 10s), via a
  shared `_within_join_grace()` helper; the recent-lines buffer still fills
  during grace via `_note_recent`. Grace state is reset in the setUp of the
  interjection/silence/floor test classes so it does not leak between tests.
  167 tests.
- 2026-08-29: Channel context (userlist + last 15 lines) is now injected for
  every persona except factual, not just chat/interject. `_system_context`'s
  guard flipped from `mode not in (chat, interject)` to `mode == FACTUAL`, so a
  serious-mood reply also gets the userlist and recent lines; factual stays
  context-free (it answers about the world, not the room). Previously serious
  was excluded on the old assumption it should not name people. 165 tests.
- 2026-08-29: The auto-interject opener now waits `JOIN_GRACE_PERIOD` (7s)
  after JOIN before firing. On join `_activity["at"]` is 0.0, so the silence
  breaker used to fire at once and call the LLM before the userlist arrived
  (the bot then invented names); `_check_silence` now bails out while
  `time.monotonic() - _joined_at < JOIN_GRACE_PERIOD`, and `main()` records
  `_joined_at` when it issues JOIN. 164 tests.
- 2026-08-29: Channel context expanded beyond the userlist. Nick status prefixes
  (+, &, @, %) are now stripped in `_parse_who_reply` / `_parse_name_reply` via a
  shared `_strip_status` helper (they are not part of the nick). The last 15
  channel lines spoken are kept in a rolling `_recent_lines` deque (maxlen=15),
  appended in the receiver for every PRIVMSG body, and injected into the chat /
  interjection context alongside the userlist (factual stays context-free):
  "Recent channel messages:\n- ...". `IDLE_PROMPT` now nudges the bot to name a
  channel user. NICK is `sloppy` and `LLM_TEMPERATURE` is 1.2 (both user-tuned).
  162 tests.
- 2026-08-29: Added terminal tracing of what the bot sends to llama.cpp: `_call_llm`
  prints `System prompt:` / `User prompt:` before each request, and the receiver
  prints `Userlist:` when the 353 NAMREPLY arrives. Cosmetic only, no behaviour
  change. 156 tests.
- 2026-08-29: Joined personas now include the channel userlist. After JOIN (once
  the 001 welcome is seen) the bot sends `WHO #hive` and records members from the
  352 (WHO) and 353 (NAMREPLY) replies into `_users["names"]`, excluding its own
  nick. `_system_context(mode)` wraps `_system_prompt(mode)` for MODE_CHAT /
  MODE_INTERJECT only, appending "The users in this IRC channel are named: a, b";
  MODE_FACTUAL is unchanged. The receiver dispatches info lines via
  `_handle_info_line` (kept out of the persona-text tests, which were trimmed to
  assert validity/structure only, so the persona can be rewritten freely).
  156 tests.
- 2026-08-29: banter/serious became a global mood instead of a per-reply mode.
  `_mood` holds the name and the time it was set; `_current_mood()` lapses
  serious back to banter once `SERIOUS_TIMEOUT` (15 min) has passed, lazily on
  read rather than on a timer thread, and logs when it does. `_set_mood` restarts
  the clock, so repeating the command extends it. `_match_mood_command` accepts
  only a line that is *nothing but* the word (bare, after the nick, or after
  `AI:`) -- "are you serious" and "be serious for once" must stay chat -- and the
  ack is sent straight from the receiver thread, as PONG already is, so it cannot
  displace a queued prompt or arrive two seconds late. The mood is applied at
  answering time in `_process_pending` via `_effective_mode`, not at capture, so
  `_pending["mode"]` still records what the message asked for and the existing
  mode tests stay meaningful. Serious gets its own persona rather than reusing
  `MODE_FACTUAL`: that one opens with a TRUE/FALSE verdict word, which is wrong
  for an ordinary question. `SERIOUS_IDLE_PROMPT` replaces "say something funny"
  for unprompted lines in serious mood -- asking a persona that was told not to
  joke for a joke reads badly either way. Boot mood is `random.choice` of the
  two, so tests that care about interjection text now pin the mood in setUp.
  Follow-up the same day: mood commands take padding when the line is aimed at
  the bot -- by nick, by `AI:`, mid-conversation or on an open floor --
  `_mood_from_words(text, loose=)` accepting the mood word plus only
  `MOOD_FILLER_WORDS` around it. That list stays short on purpose: "are you
  serious", "is it serious", "why so serious" and "stop being serious" must all
  remain ordinary chat, so none of their words are in it, and unaddressed lines
  keep the strict bare-word rule so "be serious" aimed at a human is ignored.
  Addressing itself loosened too: `_strip_lead_ins` skips up to two greetings in
  front of the nick ("hey Heretic..", "ok so Heretic") and `_strip_leading_nick`
  now eats `.!?-` as separators as well. The lead-in strip is applied only on
  the leading-nick path -- doing it to the whole message turned "hello there
  Heretic" into the prompt "there". 144 tests.
- 2026-08-29: Factchecking became a third mood and the switches announce
  themselves in the channel's own words (Alexander: "Ok I'll be serious for a
  while" / "Oh you want bants huh? Fine" / "Factchecking engaged"). `MOOD_WORDS`
  maps the command words (including "factchecking") to moods, `MOOD_MODES` maps
  a mood to the persona it answers in, and `SERIOUS_TIMEOUT` became
  `MOOD_TIMEOUT` now that two moods lapse. `_random_mood` deliberately keeps
  factchecking out of the boot draw. Two bugs fell out of the tests: bare
  "Heretic: factcheck" was parsed as a factcheck of nothing (`_match_trigger`
  returns None on an empty prompt), so `_match_mood_command` now retries against
  the message with the nick stripped; and `_split_prefix` matched "factchecking"
  as "factcheck" + the prompt "ing", so a prefix that ends in a letter now needs
  a word boundary after it -- prefixes ending in punctuation ("ai:") do not, so
  "AI:hello" still works. The chat persona's three run-together sentences
  ("normYou", "moralisticYour", "correct.Answer") were fixed with spaces and
  full stops; wording untouched. 152 tests.
- 2026-08-28: Renamed bottest2 -> irc_llm_bot and moved to ~/AI/irc_llm_bot.
  Git history moved with the directory (nothing re-created). `.qa-venv` was
  deleted rather than moved -- it embedded the old absolute path in
  `bin/activate` -- and check.sh rebuilds it on the next run. Only AGENTS.md and
  NOTES.md referenced the old name. `test_window_is_25_seconds` was pinning a
  hand-tuned knob and broke when FOLLOWUP_WINDOW was set to 35; it now asserts a
  plausible range instead of one value. 112 tests.
- 2026-08-28: Silence breaker + open floor. `_check_silence` runs from the poll
  loop; after 30 min with no channel line it queues an interjection (same
  banter/joke split) and opens a 60s window in which anything from anyone is
  answered, capped at 8 prompts. `_note_activity` fires on every PRIVMSG, so any
  chatter resets the clock, and the clock is reset before queueing so it cannot
  re-fire on the next poll. The cap deliberately covers follow-up-window
  engagements too: the first version let the first replier fall into a 25s
  conversation and escape the budget entirely, so the test asking for 8 got 9+.
  Explicit triggers are never capped -- otherwise the bot goes deaf to direct
  questions for the rest of the minute. "shut up" closes the floor. 111 tests.
- 2026-08-28: Follow-up window 15s -> 25s, and added unprompted interjections.
  `_note_chatter` counts channel lines that were not addressed to the bot; at 20
  it queues a prompt, `IDLE_REACT_CHANCE` (0.5) of the time the last line spoken
  and otherwise `IDLE_PROMPT` ("tell us something funny!"). The counter resets
  whenever the bot is engaged, and an interjection deliberately does NOT open a
  follow-up window -- nobody addressed it, so latching onto whoever spoke last
  would be intrusive. It also yields to a real prompt already queued rather than
  overwriting it. Unprompted lines use `MODE_INTERJECT`: the chat persona plus a
  clause steering to banter first, a joke every so often, and an explicit
  invitation to keep the odd unhinged non sequitur (Alexander, 2026-08-28: the
  random weird tangents are wanted, do not force them out). 98 tests.
- 2026-08-28: Added a factual answering mode. `factcheck`, `science:` and
  `research:` select a fact-checker system prompt instead of the channel
  persona; `_match_trigger` now returns (mode, prompt). A leading nick may be
  followed by a mode prefix, so "Heretic, factcheck X" works. `science` and
  `research` require their colon -- they are ordinary words and would otherwise
  fire on normal chat -- while `factcheck` stays colon-optional as before.
  First version of the prompt put a verdict word on questions too ("research:
  who discovered penicillin" -> "FALSE: Howard Florey"), so it now makes the
  CLAIM vs QUESTION distinction explicit: measured 8/8 correct after the change
  (4 questions answered plain, 4 claims opening TRUE/FALSE). Follow-ups inside
  the conversation window return to chat mode, so one factcheck does not make
  the whole conversation factual. 84 tests.
