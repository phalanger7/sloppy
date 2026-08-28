# irc_llm_bot — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 95 tests pass, quality gate clean. Two answering modes: chat (channel persona) and factual (`factcheck`, `science:`, `research:` — verdict word on claims, plain answer on questions, no jokes). A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (25s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. After `SILENCE_TIMEOUT` (30 min) with nobody talking it breaks the silence and opens the floor for `OPEN_FLOOR_WINDOW` (60s), answering anything from anyone up to `OPEN_FLOOR_MAX_PROMPTS` (8). After `IDLE_INTERJECT_AFTER` (20) unaddressed channel lines the bot chimes in unprompted, 50/50 between reacting to the last line and being asked for `IDLE_PROMPT`, under a `MODE_INTERJECT` prompt that leans to banter, tells a joke now and then, and deliberately leaves room for random tangents. LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise.

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
- 2026-08-28: Looser addressing + follow-up conversations. The nick now matches
  at the end of a sentence too ("whats the weather like, Heretic?"), with a word
  boundary check so "esoteric" does not match and a punctuation-only remainder
  ("Heretic?") is not treated as an empty prompt. After the bot replies to
  someone, `_conversation` keeps a 15s window in which anything that person says
  is treated as addressed to it, refreshed on each reply (started from the reply,
  not from their message, since generation takes seconds). "shut up" anywhere at
  the START of a prompt answers "Fine i'll shut up" and closes the window without
  calling the LLM -- anchored, so "what does shut up mean in japanese" is still a
  question. System prompt retuned from hostile toward funny: "smartarse, not its
  bully", edge aimed at the situation rather than the speaker. 68 tests.
