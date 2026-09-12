#!/usr/bin/env python3
"""Persistent per-chatter profiles.

What the bot remembers about the individual people in the channel, as opposed
to the channel-wide rolling summary in :mod:`summarizer`. Each profile holds
the last :data:`PROFILE_LINES` substantive lines that person spoke, plus room
for distilled highlights and verbatim quotes that a later pass fills in.

Identity, not nicks
-------------------
People change nick. ``Probe`` becomes ``Probe_afk`` and is still the same
person, so a profile is keyed by a stable id and carries every nick that person
has been seen under. A witnessed ``NICK`` event links the two through
:meth:`ProfileStore.link`; because the aliases persist, a link only has to be
witnessed once and holds for every session after it. The name the bot uses for
them is whichever alias they have spoken under most -- someone who is ``Probe``
for a thousand lines and ``Probe_afk`` for twenty is called Probe.

Nicks are matched case-insensitively (IRC's own rule, and the server may echo a
different casing than the one typed), while the casing last seen for each alias
is kept so the bot writes people's names the way they write them.

Threading
---------
A store is NOT internally locked. The bot mutates it from the receiver thread
under its own ``_prompt_lock`` and writes it from the summarizer worker: take a
:meth:`ProfileStore.snapshot` under that lock, then hand it to :func:`write`
outside the lock, so a disk write never blocks a reply.

Times are wall-clock ``time.time()`` epochs, not the monotonic clock used for
the bot's timers: these outlive the process that wrote them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# How many of a person's most recent substantive lines are kept. This is the
# raw material a later highlighting pass reads; it is not what gets injected
# into a prompt.
PROFILE_LINES = 25
# Caps on the distilled fields. Filled by a later pass; declared here so the
# shape of a profile lives in one place.
MAX_HIGHLIGHTS = 5
MAX_QUOTES = 3
# A profile nobody has seen for this long is dropped, and the store never keeps
# more than MAX_PROFILES people -- oldest first -- so a busy public channel
# cannot grow the file without limit.
PRUNE_AFTER_DAYS = 90
MAX_PROFILES = 200
# Bumped when the on-disk shape changes. A file from a version this code does
# not understand is ignored rather than half-read.
STORE_VERSION = 1


def default_path() -> Path:
    """Where the profile store lives: $XDG_DATA_HOME/sloppy/profiles.json."""
    root = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(root) / "sloppy" / "profiles.json"


def _key(nick: str) -> str:
    """The lookup form of a nick. IRC nicks are case-insensitive."""
    return nick.strip().lower()


def _new_profile(now: float) -> dict[str, Any]:
    """An empty profile. Every field the rest of the code reads exists here, so
    a profile is never half-shaped."""
    return {
        "aliases": {},        # nick key -> how many lines spoken under it
        "casing": {},         # nick key -> the casing last seen for it
        "lines": [],          # [[epoch, text], ...], oldest first, capped
        "highlights": [],     # filled by a later distilling pass
        "quotes": [],         # verbatim only; filled by a later pass
        "first_seen": now,
        "last_seen": now,
        "line_count": 0,             # every line ever, not just the kept ones
        "highlights_at": 0.0,        # when the distilling pass last ran
        "lines_since_highlights": 0,  # its trigger counter
    }


class ProfileStore:
    """The people the bot has met, keyed by a stable id rather than by nick.

    Not thread-safe by design -- see the module docstring.
    """

    def __init__(self) -> None:
        # id -> profile. The id is the key form of the first nick that person
        # was seen under; it never changes, so a rename does not rewrite the
        # store.
        self._profiles: dict[str, dict[str, Any]] = {}
        # Every known nick key -> the id it belongs to. Rebuilt from the
        # aliases on load rather than persisted, so it cannot fall out of step
        # with them.
        self._index: dict[str, str] = {}

    # -- identity ----------------------------------------------------------
    def _rebuild_index(self) -> None:
        self._index = {
            alias: pid
            for pid, profile in self._profiles.items()
            for alias in profile["aliases"]
        }

    def id_for(self, nick: str) -> str | None:
        """The profile id `nick` belongs to, or None if we have never seen it."""
        return self._index.get(_key(nick))

    def primary_nick(self, nick: str) -> str:
        """What to call this person: the alias they have spoken under most.

        Falls back to `nick` as given when we have never seen them, so a caller
        can use this unconditionally.
        """
        profile = self.get(nick)
        if not profile or not profile["aliases"]:
            return nick
        top = max(profile["aliases"], key=lambda a: profile["aliases"][a])
        return profile["casing"].get(top, top)

    def get(self, nick: str) -> dict[str, Any] | None:
        """The profile `nick` belongs to, under any of their aliases, or None."""
        pid = self.id_for(nick)
        return self._profiles[pid] if pid else None

    # -- writing -----------------------------------------------------------
    def note_line(self, nick: str, text: str, now: float | None = None) -> None:
        """Record that `nick` said `text`, creating their profile if needed.

        Only substantive lines belong here -- the caller filters out the noise
        the rest of the bot already ignores.
        """
        key = _key(nick)
        if not key or not text.strip():
            return
        now = time.time() if now is None else now
        pid = self._index.get(key)
        if pid is None:
            pid = key
            self._profiles[pid] = _new_profile(now)
            self._index[key] = pid
        profile = self._profiles[pid]
        profile["aliases"][key] = profile["aliases"].get(key, 0) + 1
        profile["casing"][key] = nick.strip()
        profile["lines"].append([now, text.strip()])
        del profile["lines"][:-PROFILE_LINES]
        profile["last_seen"] = now
        profile["line_count"] += 1
        profile["lines_since_highlights"] += 1

    def link(self, old: str, new: str, now: float | None = None) -> str | None:
        """Record that `old` and `new` are the same person; return their id.

        Called on a witnessed NICK change. The three cases are: only one side
        is known (the other becomes an alias of it), neither is known (a
        profile is created carrying both), or both are known separately (the
        two profiles are merged, keeping the older identity).
        """
        old_key, new_key = _key(old), _key(new)
        if not old_key or not new_key or old_key == new_key:
            return self._index.get(new_key)
        now = time.time() if now is None else now
        old_pid = self._index.get(old_key)
        new_pid = self._index.get(new_key)

        if old_pid and new_pid and old_pid != new_pid:
            # They have been talking under both names without us ever seeing
            # the change. Keep the identity we met first.
            first, second = sorted(
                (old_pid, new_pid),
                key=lambda pid: self._profiles[pid]["first_seen"],
            )
            self._merge(first, second)
            pid = first
        else:
            pid = old_pid or new_pid
            if pid is None:
                # Neither side is known: a rename is the first we have heard of
                # them, so start a profile with no lines but both names.
                pid = old_key
                self._profiles[pid] = _new_profile(now)
                self._profiles[pid]["aliases"][old_key] = 0
                self._profiles[pid]["casing"][old_key] = old.strip()

        profile = self._profiles[pid]
        profile["aliases"].setdefault(new_key, 0)
        profile["casing"][new_key] = new.strip()
        profile["last_seen"] = now
        self._index[old_key] = pid
        self._index[new_key] = pid
        return pid

    def _merge(self, keep: str, drop: str) -> None:
        """Fold the `drop` profile into `keep` and forget it."""
        into, gone = self._profiles[keep], self._profiles.pop(drop)
        for alias, count in gone["aliases"].items():
            into["aliases"][alias] = into["aliases"].get(alias, 0) + count
            self._index[alias] = keep
        into["casing"].update(gone["casing"])
        # Both line lists are oldest-first; interleave by time and re-cap.
        into["lines"] = sorted(into["lines"] + gone["lines"])[-PROFILE_LINES:]
        into["highlights"] = (into["highlights"] + gone["highlights"])[:MAX_HIGHLIGHTS]
        into["quotes"] = (into["quotes"] + gone["quotes"])[:MAX_QUOTES]
        into["first_seen"] = min(into["first_seen"], gone["first_seen"])
        into["last_seen"] = max(into["last_seen"], gone["last_seen"])
        into["line_count"] += gone["line_count"]
        into["lines_since_highlights"] += gone["lines_since_highlights"]

    def forget(self, nick: str) -> bool:
        """Erase the profile `nick` belongs to. True if there was one.

        Everything goes -- every alias of that person, their lines, and their
        highlights. Somebody asking to be forgotten is not asking to be
        remembered under their other name.
        """
        pid = self._index.pop(_key(nick), None)
        if pid is None:
            return False
        for alias in self._profiles[pid]["aliases"]:
            self._index.pop(alias, None)
        del self._profiles[pid]
        return True

    def prune(self, now: float | None = None) -> int:
        """Drop stale and surplus profiles; return how many went."""
        now = time.time() if now is None else now
        cutoff = now - PRUNE_AFTER_DAYS * 86400
        stale = [p for p, v in self._profiles.items() if v["last_seen"] < cutoff]
        for pid in stale:
            del self._profiles[pid]
        surplus = len(self._profiles) - MAX_PROFILES
        if surplus > 0:
            oldest = sorted(
                self._profiles, key=lambda pid: self._profiles[pid]["last_seen"]
            )[:surplus]
            for pid in oldest:
                del self._profiles[pid]
        else:
            surplus = 0
        if stale or surplus:
            self._rebuild_index()
        return len(stale) + surplus

    # -- reading -----------------------------------------------------------
    def known(self) -> list[dict[str, Any]]:
        """Every profile, most recently seen first, each with its display name.

        The returned dicts are the live ones -- a caller that only reads them
        (the TUI) is fine; anything that keeps one past the lock should copy.
        """
        out = []
        for pid, profile in self._profiles.items():
            top = max(profile["aliases"], key=lambda a: profile["aliases"][a],
                      default=pid)
            out.append({"id": pid, "nick": profile["casing"].get(top, top),
                        **profile})
        out.sort(key=lambda p: p["last_seen"], reverse=True)
        return out

    def snapshot(self) -> dict[str, Any]:
        """A plain, JSON-ready copy of the whole store, for :func:`write`.

        Taken under the caller's lock; the write itself happens outside it.
        """
        return {
            "version": STORE_VERSION,
            "profiles": json.loads(json.dumps(self._profiles)),
        }

    def restore(self, data: Any) -> None:
        """Replace the store's contents with a snapshot read off disk.

        Anything unrecognised is ignored rather than trusted: a profile missing
        fields is filled in from the empty shape, so an older or hand-edited
        file cannot make the rest of the code trip over a missing key.
        """
        self._profiles = {}
        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            self._index = {}
            return
        for pid, raw in (data.get("profiles") or {}).items():
            if not isinstance(pid, str) or not isinstance(raw, dict):
                continue
            profile = _new_profile(0.0)
            profile.update({k: v for k, v in raw.items() if k in profile})
            if not profile["aliases"]:
                profile["aliases"] = {pid: 1}
            profile["lines"] = [
                [t, s] for t, s in profile["lines"][-PROFILE_LINES:]
                if isinstance(s, str)
            ]
            self._profiles[pid] = profile
        self._rebuild_index()


def write(path: Path, snapshot: dict[str, Any]) -> bool:
    """Write `snapshot` to `path` atomically. True on success.

    Written to a sibling temporary file and moved into place, so a crash or a
    full disk leaves the previous store intact rather than a half-written one.
    Never raises: losing the profiles is not worth taking the bot down for.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed save must not stop the bot
        print(f"could not write {path}: {exc}", file=sys.stderr)
        return False


def read(path: Path) -> Any:
    """Read a snapshot back, or None if there is nothing usable to read.

    A missing file is the normal first-run case. A corrupt one is reported and
    treated as missing -- starting with no memory of anybody beats refusing to
    start.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - a bad file must not stop the bot
        print(f"{path} unusable: {exc}", file=sys.stderr)
        return None
