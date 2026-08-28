# bottest2 — session notes

## Current status
Bot fully operational on hive.2bd.net:#hive. JOIN waits for 001 Welcome before joining. 53 tests pass, quality gate clean. Silent acks; responds to "AI:", "factcheck", and its own nick (all case-insensitive) — the nick trigger is derived from `NICK`, so renaming the bot is a one-line change. LLM replies are reflowed into at most 3 byte-bounded PRIVMSGs. Model-side reasoning is disabled per request, and an empty completion is reported in-channel rather than swallowed. The system prompt is an in-channel persona (built from `NICK`/`CHANNEL`), deliberately crude — #hive's register is coarse and the bot should match it, not sanitise.

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
- 2026-08-28: Fixed silent empty replies. Cause: Qwen3.5 is a reasoning model;
  llama.cpp puts its <think> block in `reasoning_content`, and `max_tokens=200`
  was consumed entirely by thinking before any answer was written. Measured on
  the live server: 4 of 5 calls came back `finish_reason='length'`,
  `completion_tokens=200` (exactly the cap), `content=''`, with 678-819 chars of
  reasoning. `_call_llm` then returned `""` and the bot sent nothing, logging
  `[AI] Replied: `. Fix: request `chat_template_kwargs={"enable_thinking": false}`
  (drops replies to 16-34 completion tokens, 0/8 empty on re-test), raise the cap
  to 512 as a backstop, and raise `EmptyLLMReply` so an empty completion is
  reported in-channel instead of being silent. 31 tests.
- 2026-08-28: Capped reply length. Before: one PRIVMSG per newline with a
  450-*character* per-line cap. Measured on the live model, "explain the OSI
  model" produced 1583 chars over 23 non-blank lines -> 23 PRIVMSGs from a single
  question, and the char-based cap let an emoji-heavy line reach ~4x its byte
  budget. Now `_format_reply_lines` reflows on word boundaries into at most
  `IRC_MAX_REPLY_LINES = 3` lines, budgeted in BYTES over the full wire line
  (`IRC_MAX_LEN = 400`, incl. the "PRIVMSG #hive :" framing and CRLF), with a
  trailing ellipsis when truncated. The system prompt also asks for <=3 short
  plain-text lines, so the cap is rarely reached: the same OSI question now
  returns 486 chars in 2 PRIVMSGs, worst wire line 396 B. 38 tests.
