#!/usr/bin/env python3
"""Runtime configuration, read once from ``sloppy.toml``.

Every lever has a default in the code that asks for it, so a missing, partial
or malformed file leaves the bot running exactly as it would without one. The
file only needs to carry what you want to change -- though the one shipped
alongside this module lists everything, because a lever you cannot find is not
a lever.

TOML rather than JSON because this file exists to be edited by hand: it takes
comments, and every value in it wants a sentence saying what it does.

Values are addressed by dotted key -- ``get("chatter.followup_window", 40.0)``
reads ``followup_window`` from the ``[chatter]`` table. A value of the wrong
type is reported and ignored rather than propagated: a typo in a tuning file
should not take the bot down or, worse, silently make it behave oddly.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

CONFIG_NAME = "sloppy.toml"
# Read on top of CONFIG_NAME when it exists, and never committed. Anything that
# identifies this particular bot -- which server, which channel, which nick --
# belongs here rather than in the tracked file, so a public checkout carries
# the shape of the settings without carrying somebody's channel.
LOCAL_NAME = "sloppy.local.toml"

# Loaded contents, flat: {"chatter.followup_window": 40.0}. Empty until load()
# succeeds, which is the same as "every default applies".
_VALUES: dict[str, Any] = {}
# Problems found while loading, so the caller can surface them wherever it
# shows warnings rather than this module guessing.
_PROBLEMS: list[str] = []


def default_path() -> Path:
    """Where the config lives: beside the code, not in a data directory.

    It is part of the bot's setup rather than its state, and keeping it next to
    the source means a checkout is a working bot.
    """
    return Path(__file__).resolve().parent / CONFIG_NAME


def local_path(path: Path | None = None) -> Path:
    """The untracked overrides that sit on top of `path`."""
    base = default_path() if path is None else path
    return base.with_name(LOCAL_NAME)


def _flatten(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """One level of nesting into dotted keys; deeper tables are kept whole."""
    out: dict[str, Any] = {}
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and not prefix:
            out.update(_flatten(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def load(path: Path | None = None) -> list[str]:
    """Read the config file. Returns a list of problems, empty when all is well.

    A missing file is not a problem -- it means "use the defaults", which is a
    perfectly good way to run. Anything unreadable is reported and then ignored.
    """
    # Mutated in place rather than rebound, like the other shared containers in
    # this project -- a module-level `global` is what ruff PLW0603 is about.
    _VALUES.clear()
    _PROBLEMS.clear()
    path = default_path() if path is None else path
    # The local file is read second and wins key by key, so it only has to
    # carry what differs. A missing one is the normal case for a fresh clone.
    for source in (path, local_path(path)):
        _read_into(source)
    return list(_PROBLEMS)


def _read_into(path: Path) -> None:
    """Merge one TOML file into `_VALUES`. A missing file is not a problem."""
    try:
        with open(path, "rb") as handle:
            _VALUES.update(_flatten(tomllib.load(handle)))
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a bad config must not stop the bot
        _PROBLEMS.append(f"{path.name} unusable, using defaults: {exc}")


def get(key: str, default: Any) -> Any:
    """The configured value for `key`, or `default`.

    The default also declares the expected type. A configured value of another
    type is refused -- an int is accepted where a float is wanted, since TOML
    writes 120 and 120.0 differently and nobody should have to care.
    """
    if key not in _VALUES:
        return default
    value = _VALUES[key]
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(default, bool) != isinstance(value, bool):
        _PROBLEMS.append(f"{key}: expected {type(default).__name__}, got "
                         f"{type(value).__name__}; using {default!r}")
        return default
    if not isinstance(value, type(default)):
        _PROBLEMS.append(f"{key}: expected {type(default).__name__}, got "
                         f"{type(value).__name__}; using {default!r}")
        return default
    return value


def section(prefix: str) -> dict[str, Any]:
    """Every key directly under `prefix`, as {name: value}.

    Lets a caller enumerate something it does not know the shape of in advance
    -- the moods, above all: adding one should be a file edit, not a code
    change, so nothing may hardcode their names.
    """
    head = f"{prefix}."
    return {k[len(head):]: v for k, v in _VALUES.items() if k.startswith(head)}


def problems() -> list[str]:
    """Everything wrong with the config, including bad values found by get()."""
    return list(_PROBLEMS)


if __name__ == "__main__":
    # Print what the file actually resolves to, for checking a hand edit.
    found = load()
    print(f"config: {default_path()}")
    local = local_path()
    print(f"local:  {local}" + ("" if local.exists() else " (none)"))
    for problem in found:
        print(f"  problem: {problem}", file=sys.stderr)
    for key in sorted(_VALUES):
        print(f"  {key} = {_VALUES[key]!r}")
