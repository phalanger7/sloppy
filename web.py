#!/usr/bin/env python3
"""Fetch a web page safely and reduce it to readable text.

Standalone on purpose: nothing here imports the bot, so it can be lifted into
any other project that wants to hand a URL to an LLM. Run it directly to try a
URL from the command line.

    python3 web.py https://example.com/article

Why the fetching is ours
------------------------
Extraction libraries are not SSRF defences. trafilatura -- which this uses when
it is installed, and whose own documentation recommends "download first and
extract later" -- will happily fetch http://127.0.0.1 if you ask it to. So the
download is done here, behind the checks in :func:`check_url`, and the library
is handed a string it never went and got itself.

That matters because the caller is usually something anybody can talk to. A bot
that fetches arbitrary URLs on command is a proxy into whatever network it runs
on: the LLM server on localhost, the router's admin page, a cloud metadata
endpoint. The rules below exist to make that boring:

* http and https only -- no file:, ftp:, gopher:, data:
* the HOSTNAME IS RESOLVED and the resulting addresses are checked, because
  "internal.example.com" resolving to 127.0.0.1 defeats any name-based list
* redirects are followed by hand and every hop is re-checked, since a public
  URL is free to redirect to a private one
* the body is streamed and abandoned past a byte limit, so a huge file cannot
  be used to exhaust memory
* content types other than HTML and plain text are refused unread

There is a residual TOCTOU window: a name checked and then fetched could in
principle resolve differently for the second lookup. Closing it fully means
connecting to the checked IP and carrying the hostname in the Host header and
TLS SNI, which is a lot of machinery for an attack nobody is mounting against a
channel bot. It is written down here rather than pretended away.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import sys
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import requests

# Only these can be fetched. The rest of the URL space is either not fetchable
# over the network at all (file:, data:) or not something a channel bot has any
# business reaching (ftp:, gopher:).
ALLOWED_SCHEMES = frozenset({"http", "https"})
# Content we can turn into text. Anything else is refused without reading the
# body -- there is no point streaming a video to find out it is a video.
ALLOWED_TYPES = ("text/html", "application/xhtml", "text/plain", "text/markdown")
# Stop reading here. Enough for any article, far short of anything that would
# hurt to hold in memory.
MAX_BYTES = 2_000_000
# Per-request timeouts as (connect, read).
TIMEOUT = (5, 15)
# Redirect hops followed. Each one is re-checked.
MAX_REDIRECTS = 5
# Sent so operators can see who is asking and why, rather than a bare default.
USER_AGENT = "sloppy-irc-bot/1.0 (+article summariser; contact: channel operator)"
# The only status we read a body for. Anything else is reported as it stands.
HTTP_OK = 200
# Below this many bytes a size limit is reported in bytes rather than kB.
KB = 1000
# A CLI invocation is the program name plus one URL.
_CLI_ARGS = 2


@dataclass
class Page:
    """A fetched page, or the reason there is not one.

    `error` being set is the only thing a caller has to check; everything else
    is empty when it is.
    """

    url: str = ""
    title: str = ""
    text: str = ""
    error: str = ""

    def __bool__(self) -> bool:
        return not self.error


def _address_is_public(host: str) -> str:
    """Resolve `host` and return why it is unsafe, or "" if it is fine.

    Every address the name resolves to has to be public: a name with one public
    and one loopback address is not a name we will fetch, because which one
    gets used is not ours to decide.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return f"cannot resolve {host}: {exc}"
    if not infos:
        return f"cannot resolve {host}"
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return f"{host} resolved to something that is not an address: {raw!r}"
        # ::ffff:127.0.0.1 is loopback wearing a hat.
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_reserved
                or address.is_unspecified):
            return f"{host} resolves to {address}, which is not a public address"
    return ""


def check_url(url: str) -> str:
    """Why `url` must not be fetched, or "" if it may be.

    Kept separate from the fetching so it can be tested on its own and reused
    by anything else that takes a URL from a stranger.
    """
    try:
        parts = urlparse(url.strip())
    except ValueError as exc:
        return f"unreadable URL: {exc}"
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return f"{parts.scheme or 'that'} is not a scheme I will fetch"
    if not parts.hostname:
        return "no hostname in that URL"
    # A literal address needs no lookup, but getaddrinfo handles one and
    # returns it unchanged, so there is one path and one set of checks.
    return _address_is_public(parts.hostname)


def _strip_tags(html: str) -> str:
    """Last-resort extraction: drop the markup and keep the words.

    Used when no extraction library is installed. It keeps navigation and
    footers, so the result is noisier than trafilatura's, but it is always
    available and never wrong in a way that loses the article.
    """
    html = re.sub(r"(?is)<(script|style|noscript|template|svg)\b.*?</\1>", " ", html)
    html = re.sub(r"(?is)<!--.*?-->", " ", html)
    html = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(entity, char)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _title_of(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    return " ".join(match.group(1).split())[:200] if match else ""


def extract(html: str, url: str = "") -> tuple[str, str]:
    """Reduce fetched HTML to (title, text).

    trafilatura when it is installed, because it is markedly better at telling
    an article from the furniture around it; a tag stripper when it is not, so
    this module works on a machine where nothing has been installed.
    """
    title, text = _title_of(html), ""
    try:
        import trafilatura  # noqa: PLC0415 - optional, and only wanted here
    except ImportError:
        return title, _strip_tags(html)
    try:
        extracted = trafilatura.extract(
            html, url=url or None, favor_precision=True,
            include_comments=False, include_tables=False,
        )
        text = (extracted or "").strip()
        meta = trafilatura.extract_metadata(html)
        if meta is not None and getattr(meta, "title", None):
            title = meta.title
    except Exception:  # noqa: BLE001 - a bad page must not raise at the caller
        text = ""
    return title, text or _strip_tags(html)


def fetch(url: str, *, max_bytes: int = MAX_BYTES, timeout: tuple = TIMEOUT,
          max_redirects: int = MAX_REDIRECTS) -> Page:
    """Fetch `url` and return it as text, or a Page carrying the reason not.

    Never raises: every failure is a Page with `error` set, because the caller
    is a chat bot that has to say something either way.
    """
    current = url.strip()
    for _hop in range(max_redirects + 1):
        problem = check_url(current)
        if problem:
            return Page(url=current, error=problem)
        try:
            response = requests.get(
                current, stream=True, allow_redirects=False, timeout=timeout,
                headers={"User-Agent": USER_AGENT,
                         "Accept": "text/html,application/xhtml+xml,text/plain"},
            )
        except requests.RequestException as exc:
            return Page(url=current, error=f"could not fetch it: {exc}")
        with response:
            if response.is_redirect or response.is_permanent_redirect:
                target = response.headers.get("Location", "")
                if not target:
                    return Page(url=current, error="redirected to nowhere")
                # Resolved against the current URL, then re-checked at the top
                # of the loop: a public page may redirect to a private one.
                current = requests.compat.urljoin(current, target)
                continue
            if response.status_code != HTTP_OK:
                return Page(url=current,
                            error=f"the site returned {response.status_code}")
            kind = response.headers.get("Content-Type", "").split(";")[0].strip()
            if kind and not kind.lower().startswith(ALLOWED_TYPES):
                return Page(url=current, error=f"that is {kind}, not a page I can read")
            body = bytearray()
            for chunk in response.iter_content(8192):
                body += chunk
                if len(body) > max_bytes:
                    limit = (f"{max_bytes // KB}kB" if max_bytes >= KB
                             else f"{max_bytes} bytes")
                    return Page(url=current, error=f"the page is bigger than {limit}")
            html = body.decode(response.encoding or "utf-8", errors="replace")
        title, text = extract(html, current)
        if not text.strip():
            return Page(url=current, title=title, error="nothing readable on that page")
        return Page(url=current, title=title, text=text)
    return Page(url=current, error=f"too many redirects (over {max_redirects})")


def normalise(url: str) -> str:
    """A URL with its fragment dropped, for use as a cache key."""
    parts = urlparse(url.strip())
    return urlunparse(parts._replace(fragment=""))


if __name__ == "__main__":
    if len(sys.argv) != _CLI_ARGS:
        print(f"usage: {sys.argv[0]} <url>", file=sys.stderr)
        raise SystemExit(2)
    page = fetch(sys.argv[1])
    if not page:
        print(f"refused: {page.error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"url:   {page.url}")
    print(f"title: {page.title}")
    print(f"chars: {len(page.text)}\n")
    print(page.text[:2000])
