# sloppy

**v0.7.2**

Sloppy is a sarcastic, moody and sometimes funny AI bot for IRC. It connects to a local LLM through llama.cpp and brings a unique flavour of awkward crude humor and genuinely useful features. It understands the chat and has persistent context, making it able to chime in or roast people based on things they said earlier.

Works with any LLM running under llama.cpp. The better the model, the better the bot will work. 35B MoE models have proven to be very entertaining chatters that are able to understand the context of a chaotic chat. 9B models work fine aswell, although they're not as good at distilling the chat. It has not been tested with smaller models than 9B, but it should work, results may vary. Personalities are defined by prompts and easy to change in the configuration .toml file.

**Features include**

- Moods that randomly change to defined moods, duration of each mood can be set in config
- A rolling summary of the chat is kept to give the bot contextual awareness
- Optional logging with pattern matching relevancy calculation for longterm context
- Privacy commands allow people to ask what the bot knows about them and make them forget
- "FactCheck, Serious, Research, Science" and similar terms will make the bot respond seriously
- !image <url> analyzes an image and describes the content
- !summarize <url> summarizes a website
- !translate <text> translates words or sentences to any language (default english)
- !Quote, !Buddha gives random quotes
- !Factoid says a random (hopefully interesting or funny) factoid.
- !help displays the bot's commands
- Interjections when it just joins or when the chat is slow and can use a boost
- A TUI interface showing status, LLM calls and responses, ability to enable/disable the vision component and other settings. Includes a configuration text editor for the .toml file.
- Can run headless in tmux or as a service. Automatically detects existing tmux session and reattaches instead of starting a new instance
- Owners can be configured and can talk to the bot in PM. Optionally this can be set to *!*@*
- Owners can !purge [nick] (days) data from the bot's memory if needed
  

**Usage**

With the TUI, in a terminal:

    python3 llmbot_tui.py

Headless, with no terminal to keep open:

    python3 llmbot_core.py              # log to stdout
    python3 llmbot_core.py --log ~/.local/state/sloppy.log
    python3 llmbot_core.py --verbose    # include the full prompt dumps

As a service, which is the tidy way to leave it running -- see
`sloppy.service.example` for a systemd user unit and the commands to install
it. It restarts on failure and shuts down on SIGTERM, flushing the profiles
and the channel memory on the way out.

There is no way to attach the TUI to a bot that is already running headless:
the TUI reads the core's state in-process, and a detach/attach protocol is a
much bigger feature than this. If you want a UI you can come back to, run the
TUI under tmux -- `sloppy.sh` does it for you:

    ./sloppy.sh              start it and attach
    ./sloppy.sh --detach     start it and leave it in the background
    ./sloppy.sh --status     is it running
    ./sloppy.sh --stop       stop it

`ctrl-b d` leaves it running and gives you the terminal back; `./sloppy.sh`
again puts you back in it. Running it twice will not start a second bot on the
same channel. `SLOPPY_TMUX_SESSION` renames the session if you want more than
one.

Use tmux if you want a UI to come back to; use the systemd unit if you want
something that survives a reboot and restarts itself.

**Owner commands**

Set `[owners] masks` (see `sloppy.toml`) and those hostmasks get:

    !purge <nick>          erase them from everything the bot remembers
    !purge <nick> 3        ...but only the last 3 days of it

It clears the long-term log, the recent-line buffer and their profile, then
rebuilds the rolling summary from the lines that remain -- the summary rides in
the system message of every reply, so it is where something planted in the
bot's memory keeps working, and waiting for it to age out is not an answer.

Owners are also the only people the bot answers in a private query, and it
answers them there rather than in the channel.

NB
'bot.py' contains a very early legacy version of the bot from before it got TUI, it misses most of the features that make sloppy more than just a basic chatbot. 

Sloppy the bot mostly proudly coded itself: First versions coded by KAT Coder 2.5 and Tiel Coder. Claude was then used for quality assurance and is used for the rest of the bot's development.
