# bottest2 — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 84 tests pass, quality gate clean. Two answering modes: chat (channel persona) and factual (`factcheck`, `science:`, `research:` — verdict word on claims, plain answer on questions, no jokes). A leading nick may precede a mode prefix ("Heretic, factcheck if whales are mammals"). Silent acks; responds to "AI:", the factual prefixes, and its own nick at the START or END of a sentence (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. After answering someone, anything that person says for the next `FOLLOWUP_WINDOW` (15s, refreshed on each reply) counts as addressed to the bot; "shut up" ends it with a fixed reply and no LLM call. LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise.

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
- 2026-08-28: Switched to the Q4_K_M quant and pinned `LLM_TEMPERATURE = 1.2`.
  Swept on Q4_K_M with the persona prompt, n=9 crude + 9 factual probes per step:
  0.7 -> 2/9 crude, 1.0 -> 3/9, 1.2 -> 6/9, 1.6 -> 4/9; factual accuracy 9/9
  through 1.2 and 8/9 at 1.6, where it called TCP "Transfer Control Protocol".
  1.6 was therefore worse on BOTH axes -- the earlier crude-vs-clean numbers were
  measured at 1.6 and understated what the persona can do. Small n, so 1.2 vs 1.6
  on tone alone is inside noise, but 1.2 is at worst equal and strictly safer on
  coherence. Temperature stays an explicit per-request value, not inherited from
  the server's --temp, which gets retuned for unrelated serious work. 53 tests.
- 2026-08-28: Replaced the system prompt with an in-channel persona. The old
  "You are a helpful AI assistant ... concise and friendly" was re-censoring an
  already-abliterated model: measured on the live server, asked explicitly for
  crude output it complied 2/12 with that prompt vs 4/12 with no system prompt
  and 6/12 with a persona prompt (n=12/config, temp 0.7). Re-run at the bot's
  actual temp 1.6: old prompt 1/12, new persona 6/12. Not refusals -- zero
  refusals in any config -- but tone-softening: it answered "tell a filthy joke
  with actual swearing" with a clean eyebrows joke. The prompt is sent per
  request, so llama.cpp and other clients of the same server are unaffected.
  Caveat: compliance was scored by profanity regex, which undercounts a savage
  reply that happens to be clean; and the model is an IQ4_XS quant, where
  quantisation is known to partially restore ablated refusal directions -- an
  IQ4_NL copy exists at ~/.lmstudio/models/my-local-models/DefiantFableIQ4NL/ if
  this needs pushing further. 51 tests.
- 2026-08-28: Renamed bot to "Heretic" and made the nick trigger derive from
  `NICK` instead of the hardcoded "llmbot"/"llm-bot". Fixed two bugs found while
  doing it: (1) `receiver` gated on `ai:`/`factcheck` only, so the nick triggers
  never reached `_handle_ai_prompt` on the live wire at all — the unit tests
  called `_handle_ai_prompt` directly and so never caught it; (2) the "llmbot"
  branch sliced `message[8:]` for a 6-character prefix, which only worked when
  followed by ": ". Both trigger checks now share `_match_trigger`. 47 tests.
