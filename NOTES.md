# bottest2 — session notes

## Current status
Empty python project. Tooling is wired up (qa.toml, check.sh, AGENTS.md); no
code written yet.

## Known issues / open questions
- `./check.sh` warns that no source files matched until the first source file
  lands. That warning is correct, not a misconfiguration.
- No `.qa-baseline.txt` yet, which is the right state: with no baseline every
  finding counts as new and fails the gate. Only create one if you hit a
  finding you have consciously decided not to fix.

## Recent history (last 5 entries, oldest dropped)
- 2026-08-27: project scaffolded by projinit.
