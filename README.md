# sloppy

Sloppy is a sarcastic, moody and sometimes funny AI bot for IRC. It connects to a local LLM through llama.cpp and brings a unique flavour of awkward humor and genuinely useful features. It understands the chat and has persistent context, making it able to chime in or roast people based on things they said earlier.

Works with any LLM running under llama.cpp. The better the model, the better the bot will work. 35B MoE models have proven to be very entertaining chatters that are able to understand the context of a chaotic chat. 9B models work fine aswell, although they're not as good at distilling the chat. It has not been tested with smaller models than 9B, but it should work, results may vary. Personalities are defined by prompts and easy to change in the configuration .toml file.

**Features include**

- Moods that randomly change to defined moods, duration of each mood can be set in config
- A rolling summary of the chat is kept to give the bot contextual awareness
- Optional logging with pattern matching relevancy calculation for longterm context
- "FactCheck, Serious, Research, Science" and similar terms will make the bot respond seriously
- !image <url> analyze an image and describe the content
- !summarize <url> summarizes a website
- !translate <text> translates words or sentences to any language (default english)
- !Quote, !Buddha give random quotes
- Interjections when it just joins or when the chat is slow and can use a boost
- A TUI interface showing status, LLM calls and responses, ability to enable/disable the vision component and other settings. Includes a configuration text editor for the .toml file.

**Usage**
python3 -m llmhot_tui.py

NB
'bot.py' contains a very early legacy version of the bot from before it got TUI, it misses most of the features that make sloppy more than just a basic chatbot. 

Sloppy the bot mostly proudly coded itself: First versions coded by KAT Coder 2.5 and Tiel Coder. Claude was then used for quality assurance and is used for the rest of the bot's development.
