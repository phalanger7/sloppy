#!/usr/bin/env python3
"""Long-term channel recall: keep the whole chat log, and pull back the bit
that matters.

The rolling summary compresses everything older than the recent-line buffer
into a few hundred characters, and whatever it drops at a tick is gone for
good. This keeps the actual lines instead, on disk, and at reply time scores
them against what is being said right now -- so an argument from last Tuesday
comes back in the words people used, not as a summary of a summary.

Scoring is BM25 over one-line documents, with a recency prior and a relevance
floor. No embeddings: for a single channel the rare-term signal (Okapi's IDF)
does most of the work, and it costs a dependency and a second model to do
better. Where it falls down is paraphrase -- "that photo renaming thing" will
not match "exiftool does it in one line" -- which is what the rolling summary
is still there for. The two are meant to sit side by side.

Public surface: :class:`RecallStore`. It never raises on a bad file or a bad
record: a channel with no recall is a channel that still works.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Bumped when the shape of a record changes in a way this code cannot read.
# Carried on every record rather than in a header, because the file is appended
# to a line at a time and a header would have to be rewritten to stay true.
RECORD_VERSION = 1

# Okapi BM25's two constants, at the values the literature settles on. b scales
# the length normalisation, which is what stops a rambling line outscoring a
# short precise one purely by having more words in it; k1 saturates term
# frequency, which barely matters here because a document is one IRC line.
K1 = 1.5
B = 0.75

def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


# Where a failure reason goes. llmbot_core points this at its red warning sink.
error_sink = _stderr


def default_path(beside: Path) -> Path:
    """The log file, beside whatever other state the bot keeps."""
    return beside.with_name("chatlog.jsonl")


# A query term in more than this share of the log is treated as carrying no
# signal. IDF alone is not enough at channel scale: a corpus of a few thousand
# one-line documents leaves "the" and "out" with enough weight to outscore the
# one rare word the question was actually about. Deriving the list from the
# channel beats a hand-written one -- it correctly makes "sloppy" and "irc"
# worthless in this particular room, which no general list would know.
COMMON_TERM_RATIO = 0.08

# Words shorter than this carry no signal worth indexing. Two, not three:
# "gif" and "ssh" are exactly the kind of rare term this is for, and dropping
# them would be dropping the point.
_MIN_TERM = 2
_TERM_RE = re.compile(r"[a-z0-9']+")


def terms(text: str) -> list[str]:
    """`text` as the words the index is built on."""
    return [t for t in _TERM_RE.findall(text.lower()) if len(t) >= _MIN_TERM]


@dataclasses.dataclass(frozen=True)
class Settings:
    """The knobs a search runs under, as one thing to pass around.

    Defaults are the calibrated ones: measured against a 1500-line synthetic
    channel with five known topics, every topic was retrieved correctly and
    none wrongly for any floor between 0.1 and 0.4, with recall falling off
    above 0.5. 0.3 sits in the middle of that plateau.
    """

    # A fraction of the best a single line could score for this query, so it
    # does not need re-tuning as the log grows. See RecallStore._ideal.
    min_relevance: float = 0.3
    # Weeks, not days: halving daily put anything older than the recent-line
    # buffer out of reach, which is what this exists to reach.
    half_life_days: float = 14.0
    passages: int = 3


class RecallStore:
    """The channel's whole log, indexed for retrieval.

    Lines are held in memory and appended to disk as they arrive. Capture is
    deliberately independent of whether retrieval is switched on: turning
    recall on later against an empty log would mean waiting a week to find out
    whether it was any good.
    """

    def __init__(self, max_lines: int = 20000):
        self.max_lines = max_lines
        self._lines: list[dict[str, Any]] = []
        # Just the arrival times, for the bisect in search(). A parallel list
        # rather than a key function because bisect's key= would re-derive it
        # on every probe, on every reply.
        self._ats: list[float] = []
        # How many lines each term appears in, for IDF. Maintained as lines are
        # added rather than recomputed per query.
        self._df: Counter = Counter()
        self._total_terms = 0

    # -- capture -----------------------------------------------------------
    def add(self, nick: str, text: str, at: float | None = None) -> dict[str, Any]:
        """Record one channel line and return it as it will be stored."""
        record = {
            "v": RECORD_VERSION,
            "at": time.time() if at is None else at,
            "nick": nick,
            "text": text.strip(),
        }
        self._index(record)
        self._lines.append(record)
        self._ats.append(record["at"])
        if len(self._lines) > self.max_lines:
            self._forget_oldest(len(self._lines) - self.max_lines)
        return record

    def _index(self, record: dict[str, Any]) -> None:
        tokens = terms(record["text"])
        record["_terms"] = tokens
        self._df.update(set(tokens))
        self._total_terms += len(tokens)

    def _unindex(self, record: dict[str, Any]) -> None:
        tokens = record.get("_terms") or []
        self._df.subtract(set(tokens))
        self._total_terms -= len(tokens)

    def _forget_oldest(self, count: int) -> None:
        for record in self._lines[:count]:
            self._unindex(record)
        del self._lines[:count]
        del self._ats[:count]

    def forget(self, nicks: set[str]) -> int:
        """Drop every line by `nicks` (lowercased). Returns how many went.

        The bot promises somebody it has forgotten them; a log that still has
        their words and can quote them back next Tuesday would make that a lie.
        """
        kept = []
        dropped = 0
        for record in self._lines:
            if record["nick"].lower() in nicks:
                self._unindex(record)
                dropped += 1
            else:
                kept.append(record)
        self._lines = kept
        self._ats = [r["at"] for r in kept]
        return dropped

    def __len__(self) -> int:
        return len(self._lines)

    # -- retrieval ---------------------------------------------------------
    def _idf(self, term: str) -> float:
        """How much a term's presence is worth, by how rare it is here.

        The +0.5s are Robertson's smoothing and the outer 1+ keeps a term that
        appears in more than half the log from scoring negative -- without it a
        common word actively pushes a line down the ranking.
        """
        df = self._df.get(term, 0)
        n = len(self._lines)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _bm25(self, record: dict[str, Any], query: set[str], avgdl: float) -> float:
        tokens = record.get("_terms") or []
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if not tf:
                continue
            norm = tf + K1 * (1 - B + B * length / avgdl)
            score += self._idf(term) * (tf * (K1 + 1)) / norm
        return score

    def _discriminating(self, query: list[str]) -> set[str]:
        """The query terms rare enough here to mean anything. May be empty.

        An empty result means the question asked nothing distinctive, and the
        caller should recall nothing rather than whatever matched "the".
        """
        ceiling = max(2, int(len(self._lines) * COMMON_TERM_RATIO))
        return {t for t in set(query) if 0 < self._df.get(t, 0) <= ceiling}

    def _ideal(self, query: list[str]) -> float:
        """What a document scores if it carries every query term, once, at the
        average length.

        Raw BM25 is not comparable between channels or across time: IDF grows
        with the log, so a floor tuned at a thousand lines is a different floor
        at twenty thousand. Dividing by this turns the score into a fraction of
        the best a one-line document could do for this query, which is stable
        as the log grows and is a number a person can actually pick a value
        for.
        """
        # tf = 1 at length = avgdl, so the BM25 denominator is (1 + K1).
        return sum(self._idf(t) for t in query) * (K1 + 1) / (1 + K1)

    def search(
        self,
        query: str,
        settings: Settings | None = None,
        *,
        before: float | None = None,
        now: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        """The passages worth showing the model, best first, or [].

        `before` excludes everything from that moment on: those lines are
        already in the prompt verbatim, and recalling what sits three
        paragraphs below it is not recall. A cutoff by time rather than by
        count because the log spans restarts while the prompt's recent block
        does not -- the two are not the same tail of the same list.

        Each hit is returned with the line either side of it, and overlapping
        hits are merged into one passage. A matching line on its own is
        routinely unintelligible -- three lines of context read far better at
        the same token cost, because the neighbours are usually what the match
        was actually about.

        Returning nothing is a normal outcome and the intended one when the
        conversation has no past. `min_relevance` is the knob that decides it,
        as a fraction of the best a single line could score for this query (see
        _ideal): an interjection with no earlier context is fine, one dragging
        in an unrelated argument from Tuesday is worse than useless.
        """
        settings = Settings() if settings is None else settings
        # The log is chronological, so a time cutoff is a prefix and the pool's
        # indices still line up with self._lines -- which _passages relies on
        # to reach the neighbours either side of a hit.
        end = (bisect.bisect_left(self._ats, before)
               if before is not None else len(self._lines))
        pool = self._lines[:end]
        if not pool:
            return []
        wanted = self._discriminating(terms(query))
        ideal = self._ideal(wanted) if wanted else 0.0
        if not ideal:
            return []
        avgdl = (self._total_terms / len(self._lines)) or 1.0
        now = time.time() if now is None else now
        scored = []
        for i, record in enumerate(pool):
            score = self._bm25(record, wanted, avgdl)
            if not score:
                continue
            # Last night beats last month at equal term overlap. A half-life
            # in weeks, not the day it started as: halving daily put anything
            # older than the recent-line buffer out of reach, which is the
            # whole thing this is for.
            age_days = max(0.0, (now - record["at"]) / 86400)
            relevance = (
                (score / ideal) * 0.5 ** (age_days / settings.half_life_days)
            )
            if relevance >= settings.min_relevance:
                scored.append((relevance, i))
        if not scored:
            return []
        scored.sort(reverse=True)
        return self._passages([i for _, i in scored], settings.passages)

    def _passages(self, hits: list[int], limit: int) -> list[list[dict[str, Any]]]:
        """Hit indices as merged, in-order passages of up to `limit` passages.

        Ranking order decides which hits survive the limit; the passages
        themselves come back oldest-first, because a conversation read out of
        order is not a conversation.
        """
        spans: list[tuple[int, int]] = []
        for i in hits:
            span = (max(0, i - 1), min(len(self._lines) - 1, i + 1))
            touching = [
                s for s in spans if s[0] <= span[1] + 1 and span[0] <= s[1] + 1
            ]
            if not touching and len(spans) >= limit:
                continue
            # Folding into passages already taken costs nothing extra, so it is
            # never refused by the limit -- and the union is re-merged against
            # everything, because joining two spans can bridge a third.
            for s in touching:
                spans.remove(s)
            spans.append((min([span[0], *[s[0] for s in touching]]),
                          max([span[1], *[s[1] for s in touching]])))
        spans.sort()
        return [
            [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in self._lines[a:b + 1]
            ]
            for a, b in spans
        ]

    # -- persistence -------------------------------------------------------
    def append_to(self, path: Path, record: dict[str, Any]) -> bool:
        """Append one record to the log on disk. True on success.

        A line at a time, opened and closed each time: IRC volume makes the
        cost irrelevant, and a held handle is a held handle to lose on a crash.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                json.dump(
                    {k: v for k, v in record.items() if not k.startswith("_")},
                    handle, ensure_ascii=False,
                )
                handle.write("\n")
            return True
        except Exception as exc:  # noqa: BLE001 - losing a line is not fatal
            error_sink(f"could not append to {path}: {exc}")
            return False

    def rewrite(self, path: Path) -> bool:
        """Write the whole log out again, atomically. True on success.

        For the two cases that cannot be an append: trimming to `max_lines` at
        startup, and erasing somebody who asked to be forgotten.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                for record in self._lines:
                    json.dump(
                        {k: v for k, v in record.items() if not k.startswith("_")},
                        handle, ensure_ascii=False,
                    )
                    handle.write("\n")
            os.replace(tmp, path)
            return True
        except Exception as exc:  # noqa: BLE001 - a failed write must not stop the bot
            error_sink(f"could not write {path}: {exc}")
            return False

    def load(self, path: Path) -> tuple[int, int]:
        """Read the log back. Returns (lines kept, lines dropped as unusable).

        A record that is not a usable line is skipped rather than trusted, and
        one written by a newer version is skipped too. A missing file is the
        normal first run. Anything left over `max_lines` is trimmed here, which
        is the only moment it is worth paying for -- the caller rewrites the
        file when this says something was dropped.
        """
        self._lines = []
        self._ats = []
        self._df = Counter()
        self._total_terms = 0
        bad = 0
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.readlines()
        except FileNotFoundError:
            return 0, 0
        except Exception as exc:  # noqa: BLE001 - an unreadable log is not fatal
            error_sink(f"{path} unusable: {exc}")
            return 0, 0
        for raw_line in raw:
            line = raw_line.strip()
            if not line:
                continue
            record = _read_record(line)
            if record is None:
                bad += 1
                continue
            self._index(record)
            self._lines.append(record)
            self._ats.append(record["at"])
        if len(self._lines) > self.max_lines:
            over = len(self._lines) - self.max_lines
            self._forget_oldest(over)
            bad += over
        return len(self._lines), bad


def _read_record(line: str) -> dict[str, Any] | None:
    """One log line as a usable record, or None if it is not one."""
    try:
        record = json.loads(line)
    except Exception:  # noqa: BLE001 - a truncated tail is expected after a crash
        return None
    if not isinstance(record, dict) or record.get("v") != RECORD_VERSION:
        return None
    at, nick, text = record.get("at"), record.get("nick"), record.get("text")
    if not isinstance(at, int | float) or not isinstance(nick, str):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return {"v": RECORD_VERSION, "at": float(at), "nick": nick, "text": text}
