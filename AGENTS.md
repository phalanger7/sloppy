# irc_llm_bot

## Validation
- Fast loop, after every edit:  ./check.sh --fast
- Before calling anything done: ./check.sh   (paste the output)

`./check.sh` is a ratchet, not a pass/fail gate: it fails when a count *rises*
or a new finding appears. Never run `--update-baseline` to make a failure go
away — that accepts the finding permanently. Fix it, or say why it should be
accepted and ask first.

- Which changed lines no test executes: `covgap`
- A test that passes alone but fails in the suite: `flakehunt --repeat 10 --test <id>`, then `flakehunt --bisect <id>`

Coverage shows a line was *executed*, not that anything asserted on it. Do not
call a line tested because it is covered.

## Working on this project
When something is reported broken, follow the `diagnose` skill from Step 1. Do
not start by reading code looking for something suspicious.

For design decisions with no single correct answer: settle on a reasonable
choice yourself and state it in one line. If you genuinely cannot decide with
confidence after one attempt — especially before running simulations, trying
several alternatives, or spending extended reasoning on it — use the
`ask_user_question` tool to check with me directly. Don't keep exploring alone.

Don't endlessly re-think the same thing. Draw a conclusion once you have
gathered enough information and thought it through. If a term in a prompt is
unclear, stop and ask what was meant rather than enumerating interpretations.
