#!/usr/bin/env python3
"""Tests for Phase 2: AI prompt detection."""

import socket
import threading
import time
import unittest
from unittest import mock

import bot


class TestSend(unittest.TestCase):
    """Test the send helper."""

    def test_send_encodes_and_sends(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot.send(sock, "PRIVMSG #hive :hello")
        sock.send.assert_called_once_with(b"PRIVMSG #hive :hello\r\n")


class TestReceiverPong(unittest.TestCase):
    """Test that PING messages trigger PONG replies."""

    def test_ping_triggers_pong(self):
        sock = mock.MagicMock(spec=socket.socket)
        # Simulate a PING arriving in the buffer
        data = b"PING :abc123\r\n"

        def recv_side_effect(size):
            recv_side_effect.call_count += 1
            if recv_side_effect.call_count == 1:
                return data
            return ""  # exit after first recv
        recv_side_effect.call_count = 0
        sock.recv.side_effect = recv_side_effect

        t = threading.Thread(target=bot.receiver, args=(sock,))
        t.start()
        t.join(timeout=2)

        # Should have sent PONG
        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertIn(b"PONG :abc123\r\n", sends)


class TestReceiverDispatch(unittest.TestCase):
    """The receiver must forward every trigger, not just ai:/factcheck."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        bot._end_conversation()

    def _feed(self, text):
        sock = mock.MagicMock(spec=socket.socket)
        payload = f":gil!u@h PRIVMSG #hive :{text}\r\n".encode()
        chunks = [payload, b""]
        sock.recv.side_effect = lambda size: chunks.pop(0) if chunks else b""
        t = threading.Thread(target=bot.receiver, args=(sock,))
        t.start()
        t.join(timeout=2)
        return bot.get_pending_prompt()

    def test_receiver_forwards_nick_trigger(self):
        self.assertEqual(self._feed(f"{bot.NICK}: what is 2+2?"), "what is 2+2?")

    def test_receiver_forwards_ai_trigger(self):
        self.assertEqual(self._feed("AI: what is 2+2?"), "what is 2+2?")

    def test_receiver_forwards_factcheck_trigger(self):
        self.assertEqual(self._feed("factcheck: the sky is green"), "the sky is green")

    def test_receiver_ignores_ordinary_chat(self):
        self.assertEqual(self._feed("just talking about the weather"), "")


class TestParsePrivmsg(unittest.TestCase):
    """Test PRIVMSG parsing."""

    def test_parses_standard_privmsg(self):
        result = bot._parse_privmsg(":alice!alice@host PRIVMSG #hive :hello")
        self.assertEqual(result, ("alice", "hello"))

    def test_returns_none_for_non_privmsg(self):
        self.assertIsNone(bot._parse_privmsg("MODE #hive +o alice"))

    def test_parses_with_empty_message(self):
        result = bot._parse_privmsg(":bob!bob@host PRIVMSG #hive :")
        self.assertEqual(result, ("bob", ""))


class TestHandleAIPrompt(unittest.TestCase):
    """Test AI: prompt capture and acknowledgment."""

    def setUp(self):
        # Reset pending prompt and the follow-up window before each test
        bot._end_conversation()
        with bot._prompt_lock:
            bot._pending["prompt"] = ""

    def test_captures_prompt_silent(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "AI: what is the weather?")
        self.assertEqual(bot.get_pending_prompt(), "what is the weather?")
        self.assertEqual(sock.send.call_count, 0)

    def test_ignores_empty_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "AI:   ")
        self.assertEqual(bot.get_pending_prompt(), "")
        self.assertEqual(sock.send.call_count, 0)

    def test_strips_leading_whitespace(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "bob", "AI:   hello world")
        self.assertEqual(bot.get_pending_prompt(), "hello world")

    def test_factcheck_captures_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "bob", "factcheck: is the sky blue?")
        self.assertEqual(bot.get_pending_prompt(), "is the sky blue?")
        self.assertEqual(sock.send.call_count, 0)

    def test_case_insensitive_ai(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "carol", "Ai: hello")
        self.assertEqual(bot.get_pending_prompt(), "hello")
        self.assertEqual(sock.send.call_count, 0)

    def test_nick_captures_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "eve", f"{bot.NICK}: hello there")
        self.assertEqual(bot.get_pending_prompt(), "hello there")
        self.assertEqual(sock.send.call_count, 0)

    def test_nick_trigger_comma_separator(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "frank", f"{bot.NICK}, check this")
        self.assertEqual(bot.get_pending_prompt(), "check this")
        self.assertEqual(sock.send.call_count, 0)

    def test_case_insensitive_nick_upper(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "grace", f"{bot.NICK.upper()}: test")
        self.assertEqual(bot.get_pending_prompt(), "test")
        self.assertEqual(sock.send.call_count, 0)

    def test_case_insensitive_nick_lower(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "hank", f"{bot.NICK.lower()}: test")
        self.assertEqual(bot.get_pending_prompt(), "test")
        self.assertEqual(sock.send.call_count, 0)

    def test_bare_nick_is_not_a_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "ivy", bot.NICK)
        self.assertEqual(bot.get_pending_prompt(), "")
        self.assertEqual(sock.send.call_count, 0)

    def test_case_insensitive_factcheck(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "dave", "FACTCHECK: verify this")
        self.assertEqual(bot.get_pending_prompt(), "verify this")
        self.assertEqual(sock.send.call_count, 0)

    def test_nick_trigger_follows_the_configured_nick(self):
        """Triggers must derive from NICK, not hardcoded strings."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "gil", f"{bot.NICK}: hello there")
        self.assertEqual(bot.get_pending_prompt(), "hello there")

    def test_nick_trigger_without_punctuation(self):
        """'Heretic hello' must work, not just 'Heretic: hello'."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "gil", f"{bot.NICK} hello there")
        self.assertEqual(bot.get_pending_prompt(), "hello there")

    def test_old_hardcoded_nick_no_longer_triggers(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "gil", "llmbot: hello there")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_factcheck_prompt_not_truncated(self):
        """Prefix lengths must come from the trigger, not magic offsets."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "gil", "factcheck the moon is cheese")
        self.assertEqual(bot.get_pending_prompt(), "the moon is cheese")

    def test_get_pending_prompt_clears_after_read(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = "test question"
        first = bot.get_pending_prompt()
        second = bot.get_pending_prompt()
        self.assertEqual(first, "test question")
        self.assertEqual(second, "")


class TestTruncateForIrc(unittest.TestCase):
    """Test IRC message truncation."""

    def test_short_text_unchanged(self):
        self.assertEqual(bot._truncate_for_irc("hello"), "hello")

    def test_long_text_truncated_with_ellipsis(self):
        long_text = "x" * 500
        result = bot._truncate_for_irc(long_text)
        wire = f"PRIVMSG {bot.CHANNEL} :{result}\r\n".encode("utf-8")
        self.assertLessEqual(len(wire), bot.IRC_MAX_LEN)
        self.assertTrue(result.endswith("…"))


class TestCallLLM(unittest.TestCase):
    """Test LLM call via OpenAI SDK."""

    def test_call_llm_returns_content(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "Hello from AI!"

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as mock_create:
            result = bot._call_llm("what is 2+2?")
            self.assertEqual(result, "Hello from AI!")
            mock_create.assert_called_once()
            args = mock_create.call_args
            self.assertEqual(args.kwargs["model"], bot.LLM_MODEL)
            self.assertEqual(args.kwargs["messages"][0]["role"], "system")
            self.assertEqual(args.kwargs["messages"][1]["content"], "what is 2+2?")

    def test_call_llm_disables_model_side_reasoning(self):
        """Reasoning models must not spend the token budget on a <think> block.

        Qwen3.5 emitted 678-819 chars of reasoning_content and hit the 200-token
        cap before writing any answer, so content came back empty.
        """
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        mock_response.choices[0].finish_reason = "stop"

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as mock_create:
            bot._call_llm("tell us a joke")

        kwargs = mock_create.call_args.kwargs
        self.assertFalse(
            kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"],
            "must ask the server to skip the model's reasoning block",
        )
        self.assertGreaterEqual(
            kwargs["max_tokens"], 512,
            "token budget must leave room for an answer even if thinking is not skipped",
        )

    def test_call_llm_raises_when_reply_is_empty(self):
        """An empty completion must be an error, not a silent no-op."""
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].finish_reason = "length"

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with self.assertRaises(bot.EmptyLLMReply):
                bot._call_llm("tell us a joke")

    def test_call_llm_raises_when_content_is_none(self):
        """Some servers send content: null alongside reasoning_content."""
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].finish_reason = "length"

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with self.assertRaises(bot.EmptyLLMReply):
                bot._call_llm("tell us a joke")

    def test_call_llm_trims_whitespace(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "  spaced out  "

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            result = bot._call_llm("test")
            self.assertEqual(result, "spaced out")


class TestProcessPending(unittest.TestCase):
    """Test the pending prompt processing loop."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""

    def test_processes_and_replies(self):
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with bot._prompt_lock:
                bot._pending["prompt"] = "what is 2+2?"
            bot._process_pending(sock)

        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertIn(b"PRIVMSG #hive :The answer is 42.\r\n", sends)

    def test_short_multiline_reply_is_packed_into_one_message(self):
        """Short lines are reflowed, not sent as one PRIVMSG each."""
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "Line one.\nLine two.\nLine three."

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with bot._prompt_lock:
                bot._pending["prompt"] = "test"
            bot._process_pending(sock)

        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertEqual(sends, [b"PRIVMSG #hive :Line one. Line two. Line three.\r\n"])

    def test_long_reply_capped_at_three_messages(self):
        """A 23-PRIVMSG flood was possible before; cap it."""
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "\n".join(f"bullet number {i} " + "x" * 60 for i in range(30))

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with bot._prompt_lock:
                bot._pending["prompt"] = "test"
            bot._process_pending(sock)

        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertLessEqual(len(sends), bot.IRC_MAX_REPLY_LINES)
        self.assertTrue(sends[-1].rstrip(b"\r\n").endswith("…".encode()))


class TestAddressedAtEnd(unittest.TestCase):
    """The nick may come at the end of a sentence, not just the start."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        bot._end_conversation()

    def test_trailing_nick_with_comma_and_question_mark(self):
        self.assertEqual(
            bot._match_trigger(f"whats the weather like, {bot.NICK}?"),
            (bot.MODE_CHAT, "whats the weather like?"),
        )

    def test_trailing_nick_without_punctuation(self):
        self.assertEqual(bot._match_trigger(f"talk to me {bot.NICK}"), (bot.MODE_CHAT, "talk to me"))

    def test_trailing_nick_is_case_insensitive(self):
        self.assertEqual(
            bot._match_trigger(f"hello there {bot.NICK.upper()}"), (bot.MODE_CHAT, "hello there"))

    def test_mid_sentence_mention_is_not_addressed(self):
        self.assertIsNone(bot._match_trigger(f"i think {bot.NICK} is broken"))

    def test_bare_trailing_nick_is_not_a_prompt(self):
        self.assertIsNone(bot._match_trigger(f"{bot.NICK}?"))

    def test_word_ending_in_nick_does_not_match(self):
        self.assertIsNone(bot._match_trigger("that sounds esoteric"))


class TestFollowUpConversation(unittest.TestCase):
    """After being addressed, keep talking to that person for a short window."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        bot._end_conversation()

    def test_follow_up_from_same_person_is_picked_up(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        self.assertEqual(bot.get_pending_prompt(), "hello")
        # No trigger at all this time.
        bot._handle_ai_prompt(sock, "alice", "and what about tomorrow")
        self.assertEqual(bot.get_pending_prompt(), "and what about tomorrow")

    def test_follow_up_from_someone_else_is_ignored(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        bot._handle_ai_prompt(sock, "bob", "just chatting to alice")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_follow_up_after_the_window_is_ignored(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        with bot._prompt_lock:
            bot._conversation["deadline"] = time.monotonic() - 1
        bot._handle_ai_prompt(sock, "alice", "still there?")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_window_is_the_configured_length(self):
        sock = mock.MagicMock(spec=socket.socket)
        before = time.monotonic()
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        with bot._prompt_lock:
            remaining = bot._conversation["deadline"] - before
        self.assertAlmostEqual(remaining, bot.FOLLOWUP_WINDOW, delta=1.0)

    def test_direct_address_still_works_for_anyone(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        bot._handle_ai_prompt(sock, "bob", f"{bot.NICK}: what about me")
        self.assertEqual(bot.get_pending_prompt(), "what about me")


class TestShutUp(unittest.TestCase):
    """"shut up" ends the conversation with a fixed reply and no LLM call."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
            bot._pending["stop"] = False
        bot._end_conversation()

    def _run(self, sender, message):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, sender, message)
        with mock.patch.object(bot._llm_client.chat.completions, "create") as create:
            bot._process_pending(sock)
        return sock, create

    def test_shut_up_gets_the_fixed_reply_without_calling_the_llm(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        sock, create = self._run("alice", "shut up")
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertIn(f"PRIVMSG {bot.CHANNEL} :Fine i'll shut up\r\n".encode(), sends)
        create.assert_not_called()

    def test_shut_up_ends_the_follow_up_window(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        self._run("alice", "shut up")
        bot._handle_ai_prompt(sock, "alice", "you still awake")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_shut_up_works_when_directly_addressed(self):
        sock, create = self._run("bob", f"{bot.NICK}, shut up")
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertIn(f"PRIVMSG {bot.CHANNEL} :Fine i'll shut up\r\n".encode(), sends)
        create.assert_not_called()

    def test_shut_up_mentioned_inside_a_real_question_is_not_a_stop(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: what does shut up mean in japanese")
        self.assertEqual(bot.get_pending_prompt(), "what does shut up mean in japanese")


class TestFactualMode(unittest.TestCase):
    """factcheck / science: / research: switch to a factual, non-funny prompt."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        bot._end_conversation()

    def test_factcheck_selects_factual_mode(self):
        self.assertEqual(
            bot._match_trigger("factcheck: whales are fish"),
            (bot.MODE_FACTUAL, "whales are fish"))

    def test_factcheck_without_colon(self):
        self.assertEqual(
            bot._match_trigger("factcheck whales are fish"),
            (bot.MODE_FACTUAL, "whales are fish"))

    def test_science_prefix(self):
        self.assertEqual(
            bot._match_trigger("science: why is the sky blue"),
            (bot.MODE_FACTUAL, "why is the sky blue"))

    def test_research_prefix(self):
        self.assertEqual(
            bot._match_trigger("research: who first sequenced DNA"),
            (bot.MODE_FACTUAL, "who first sequenced DNA"))

    def test_nick_followed_by_factcheck(self):
        """'heretic, factcheck if whales are mammals'"""
        self.assertEqual(
            bot._match_trigger(f"{bot.NICK}, factcheck if whales are mammals"),
            (bot.MODE_FACTUAL, "if whales are mammals"))

    def test_nick_followed_by_science(self):
        self.assertEqual(
            bot._match_trigger(f"{bot.NICK}: science: why is the sky blue"),
            (bot.MODE_FACTUAL, "why is the sky blue"))

    def test_nick_alone_is_still_chat(self):
        self.assertEqual(
            bot._match_trigger(f"{bot.NICK}: tell me a joke"),
            (bot.MODE_CHAT, "tell me a joke"))

    def test_bare_science_word_is_not_a_trigger(self):
        """'science' needs its colon; otherwise ordinary chat would trigger it."""
        self.assertIsNone(bot._match_trigger("science is great and you know it"))

    def test_bare_research_word_is_not_a_trigger(self):
        self.assertIsNone(bot._match_trigger("research shows that irc is dead"))

    def test_factual_prompt_differs_from_chat_prompt(self):
        self.assertNotEqual(
            bot._system_prompt(bot.MODE_FACTUAL), bot._system_prompt(bot.MODE_CHAT))

    def test_factual_prompt_asks_for_a_verdict(self):
        prompt = bot._system_prompt(bot.MODE_FACTUAL)
        self.assertIn("TRUE", prompt)
        self.assertIn("FALSE", prompt)

    def test_factual_prompt_is_not_trying_to_be_funny(self):
        prompt = bot._system_prompt(bot.MODE_FACTUAL).lower()
        for banned in ("funny", "smartarse", "swear", "dark humour", "crude"):
            self.assertNotIn(banned, prompt)

    def test_factual_prompt_keeps_the_irc_length_constraint(self):
        self.assertIn("3 short lines", bot._system_prompt(bot.MODE_FACTUAL))

    def test_call_llm_sends_the_factual_prompt(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "TRUE. Whales are mammals."
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("if whales are mammals", bot.MODE_FACTUAL)
        self.assertEqual(
            create.call_args.kwargs["messages"][0]["content"],
            bot._system_prompt(bot.MODE_FACTUAL))

    def test_mode_survives_the_pending_queue(self):
        """The mode captured by the receiver must reach the LLM call."""
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "TRUE."
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}, factcheck if whales are mammals")
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._process_pending(sock)
        self.assertEqual(
            create.call_args.kwargs["messages"][0]["content"],
            bot._system_prompt(bot.MODE_FACTUAL))

    def test_follow_up_returns_to_chat_mode(self):
        """A factcheck does not put the whole conversation into factual mode."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "factcheck whales are fish")
        bot.get_pending_prompt()
        bot._handle_ai_prompt(sock, "alice", "and what about dolphins")
        with bot._prompt_lock:
            self.assertEqual(bot._pending["mode"], bot.MODE_CHAT)


class TestSystemPrompt(unittest.TestCase):
    """The persona must follow the bot's identity and stay uncensored."""

    def test_uses_configured_nick_and_channel(self):
        original = bot.NICK
        try:
            bot.NICK = "Zaphod"
            prompt = bot._system_prompt()
            self.assertIn("Zaphod", prompt)
            self.assertIn(bot.CHANNEL, prompt)
            self.assertNotIn("Heretic", prompt)
        finally:
            bot.NICK = original

    def test_does_not_carry_assistant_framing(self):
        """The framing that measurably re-censored the model must stay out."""
        prompt = bot._system_prompt().lower()
        for banned in ("helpful ai assistant", "helpful assistant", "friendly"):
            self.assertNotIn(banned, prompt)

    def test_keeps_the_irc_length_constraint(self):
        self.assertIn("3 short lines", bot._system_prompt())

    def test_temperature_is_sent_explicitly(self):
        """The bot pins its own temperature; the server's --temp is retuned for
        other models and must not leak into the channel's persona."""
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hello")
        self.assertEqual(create.call_args.kwargs["temperature"], bot.LLM_TEMPERATURE)

    def test_temperature_stays_in_the_coherent_range(self):
        """Above ~1.2 the model started getting facts wrong; below it went tame."""
        self.assertGreaterEqual(bot.LLM_TEMPERATURE, 1.0)
        self.assertLessEqual(bot.LLM_TEMPERATURE, 1.2)

    def test_is_sent_with_every_request(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hello")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], bot._system_prompt())


class TestFormatReplyLines(unittest.TestCase):
    """Reply reflow: byte-bounded, message-count-bounded."""

    def test_short_reply_is_one_line(self):
        self.assertEqual(bot._format_reply_lines("Paris"), ["Paris"])

    def test_never_exceeds_max_reply_lines(self):
        text = " ".join(f"word{i}" for i in range(5000))
        self.assertLessEqual(len(bot._format_reply_lines(text)), bot.IRC_MAX_REPLY_LINES)

    def test_every_line_fits_the_byte_budget(self):
        """Property: for any reply, every emitted wire line fits in the IRC limit."""
        cases = [
            "a" * 5000,
            " ".join(["hello"] * 900),
            "🧀" * 900,                      # 4 bytes per char
            "Paris 🇫🇷 " * 300,
            "supercalifragilistic" * 400,    # no spaces to break on
            "line\n" * 900,
        ]
        for text in cases:
            with self.subTest(text=text[:20]):
                for line in bot._format_reply_lines(text):
                    wire = f"PRIVMSG {bot.CHANNEL} :{line}\r\n".encode("utf-8")
                    self.assertLessEqual(len(wire), bot.IRC_MAX_LEN)

    def test_truncation_is_marked(self):
        lines = bot._format_reply_lines(" ".join(["word"] * 5000))
        self.assertTrue(lines[-1].endswith("…"))

    def test_no_blank_lines_emitted(self):
        for line in bot._format_reply_lines("one\n\n\n   \n\ntwo"):
            self.assertTrue(line.strip())

    def test_empty_input_yields_nothing(self):
        self.assertEqual(bot._format_reply_lines("   \n  "), [])

    def test_handles_llm_error_gracefully(self):
        sock = mock.MagicMock(spec=socket.socket)

        with mock.patch.object(
            bot._llm_client.chat.completions, "create",
            side_effect=Exception("connection refused")
        ):
            with bot._prompt_lock:
                bot._pending["prompt"] = "broken prompt"
            bot._process_pending(sock)

        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertTrue(any(b"LLM error" in s for s in sends))

    def test_empty_reply_is_reported_not_silent(self):
        """The reported bug: bot logged "[AI] Replied: " and said nothing in channel."""
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].finish_reason = "length"

        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with bot._prompt_lock:
                bot._pending["prompt"] = "tell us a joke"
            bot._process_pending(sock)

        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertTrue(sends, "bot must say something rather than go silent")
        self.assertTrue(any(b"PRIVMSG #hive :" in s for s in sends))

    def test_noop_when_no_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._process_pending(sock)
        self.assertEqual(sock.send.call_count, 0)


class TestBotConstants(unittest.TestCase):
    """Test that constants are set correctly."""

    def test_server(self):
        self.assertEqual(bot.SERVER, "hive.2bd.net")

    def test_channel(self):
        self.assertEqual(bot.CHANNEL, "#hive")

    def test_nick(self):
        self.assertEqual(bot.NICK, "Heretic")


if __name__ == "__main__":
    unittest.main()
