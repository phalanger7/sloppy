Sloppy is an AI bot for IRC. It connects to a local LLM and brings a uniquely sarcastic, humorous and mildly awkward 
personality to the chat. It knows who the chatters are and understands the chat and will chime in or roast people based 
on things they said in the chat.

Works with any LLM running under llama.cpp. The better the model, the better the bot will work. 35B MoE models have proven to be very entertaining chatters that are able to understand the context of a chaotic chat. 9B models work fine aswell, although they're not as good at distilling the chat. It has not been tested with smaller models than 9B, but it should work, just dont expect a 1.5B model to be a great conversationalist.

Features include
- Moods
- FactCheck and Science function to get serious answers
- Image recognition: Sloppy can analyze image links and describe the contents and comment on it
- Interjections when it just joins or when the chat is slow and can use a boost
- 
- a TUI interface showing LLM calls and responses, ability to enable/disable the vision component, ...

Todo:
- More sophisticated mood system
- Persistent memory
- Building profiles of regular chatters
- Summarize chat and identify interesting quotes for better responses


Sloppy the bot proudly coded itself: First versions coded by KAT Coder 2.5, subsequent updates coded by Tiel Coder.
