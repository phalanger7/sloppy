# irc_llm_bot — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 95 tests pass, quality gate clean. Two answering modes: chat (channel persona) and factual (`factcheck`, `science:`, `research:` — verdict word on claims, plain answer on questions, no jokes). A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (25s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. After `SILENCE_TIMEOUT` (30 min) with nobody talking it breaks the silence and opens the floor for `OPEN_FLOOR_WINDOW` (60s), answering anything from anyone up to `OPEN_FLOOR_MAX_PROMPTS` (8). After `IDLE_INTERJECT_AFTER` (20) unaddressed channel lines the bot chimes in unprompted, 50/50 between reacting to the last line and being asked for `IDLE_PROMPT`, under a `MODE_INTERJECT` prompt that leans to banter, tells a joke now and then, and deliberately leaves room for random tangents. LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise. On top of the per-reply modes there are three global moods: `banter`, `serious` and `factcheck`, switched by the bare word (`serious`, `Heretic: factcheck`, `AI: banter`) and announced in-channel without an LLM call ("Ok I'll be serious for a while", "Oh you want bants huh? Fine", "Factchecking engaged"). Serious and factchecking swap the chat and interjection personas for `MODE_SERIOUS` / `MODE_FACTUAL` (`MOOD_MODES`) until someone names another mood or `MOOD_TIMEOUT` (15 min) passes; banter is the resting state and never expires. `factcheck <claim>` is still the one-off it always was, and any message that names a mode itself ignores the mood. The boot mood is a coin flip between banter and serious (`_random_mood`) — never factchecking. Addressed to the bot, a mood command may carry filler ("Heretic, be serious for once"), unaddressed only the bare word counts. Addressing also tolerates a greeting before the nick ("hey Heretic.. whats up") and any of `:,;.!?-` after it.

## Known issues / open questions
- Uses raw TCP (not `irc` lib) due to Python 3.14 incompatibility with `tempora` dependency.
- No PING/PONG handling yet — may time out on long idle. (The receiver does answer
  PING; this note is about idle timeouts, not parsing.)
- `LLM_MODEL` is still `"llama-3.2-3b-instruct"` while the server actually serves
  `qwen35-9b`. llama.cpp ignores the field, so this is cosmetic, but it is
  misleading and worth correcting.
- `IRC_MAX_LEN = 400` assumes a worst-case ~100-byte server hostmask prefix. It
  has not been measured against hive.2bd.net's actual prefix length.
- `LLM_TEMPERATURE` is pinned at 1.2 per request. Intended direction (Alexander,
  2026-08-28) is to eventually drop the parameter and inherit the server's
  `--temp` instead; kept explicit for now so the persona does not drift when the
  server is retuned for other models. Dropping it also retires
  `test_temperature_stays_in_the_coherent_range`.

## Recent history (last 5 entries, oldest dropped)
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
