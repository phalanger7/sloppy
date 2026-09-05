# sloppy

Sloppy is an AI bot for IRC. It connects to a local LLM and brings a uniquely sarcastic, humorous and mildly awkward 
personality to the chat. It knows who the chatters are and understands the chat and will chime in or roast people based 
on things they said in the chat.

Works with any LLM running under llama.cpp. The better the model, the better the bot will work. 35B MoE models have proven to be very entertaining chatters that are able to understand the context of a chaotic chat. 9B models work fine aswell, although they're not as good at distilling the chat. It has not been tested with smaller models than 9B, but it should work, just dont expect a 1.5B model to be a great conversationalist.

Features include
- Moods
- A rolling summary of the chat is kept to give the bot contextual awareness
- FactCheck, Research and Science function to get serious answers
- Image recognition: Sloppy can analyze image links and describe the contents and comment on it (!img <url> or 'whats in the picture Probe posted?'
- Interjections when it just joins or when the chat is slow and can use a boost
- A TUI interface showing status, LLM calls and responses, ability to enable/disable the vision component and more

Todo:
- More sophisticated mood system
- Persistent memory between sessions
- Keeping highlights of regular chatters



Sloppy the bot proudly coded itself: First versions coded by KAT Coder 2.5, subsequent updates coded by Tiel Coder.
