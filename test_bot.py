#!/usr/bin/env python3
"""Tests for Phase 2: AI prompt detection."""

import random
import socket
import threading
import time
import unittest
from unittest import mock

import bot
import llmbot_core


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

    def setUp(self):
        # Recent channel history is injected into the messages list (between the
        # system prompt and the user's message), so clear it here to keep
        # messages[1] from leaking a line a prior test left behind.
        with bot._prompt_lock:
            bot._recent_lines.clear()
            bot._recent_senders.clear()

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


class TestLeadInAddressing(unittest.TestCase):
    """A greeting before the nick still addresses the bot."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        bot._end_conversation()
        bot._close_open_floor()

    def test_greeting_before_the_nick(self):
        self.assertEqual(
            bot._match_trigger(f"hey {bot.NICK}.. whats up"),
            (bot.MODE_CHAT, "whats up"))

    def test_greetings_and_separators(self):
        for message, prompt in (
            (f"hey {bot.NICK}, whats up", "whats up"),
            (f"yo {bot.NICK} whats up", "whats up"),
            (f"hi {bot.NICK}: whats up", "whats up"),
            (f"ok so {bot.NICK} whats up", "whats up"),
            (f"{bot.NICK}... whats up", "whats up"),
            (f"{bot.NICK} - whats up", "whats up"),
        ):
            with self.subTest(message=message):
                self.assertEqual(bot._match_trigger(message),
                                 (bot.MODE_CHAT, prompt))

    def test_lead_in_survives_into_the_pending_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"hey {bot.NICK}.. whats up")
        self.assertEqual(bot.get_pending_prompt(), "whats up")

    def test_lead_in_still_reaches_a_mode_prefix(self):
        self.assertEqual(
            bot._match_trigger(f"hey {bot.NICK}, factcheck whales are fish"),
            (bot.MODE_FACTUAL, "whales are fish"))

    def test_an_ordinary_word_before_the_nick_is_not_addressing(self):
        """Only greetings are skipped; anything else is talking *about* it."""
        for message in (f"apparently {bot.NICK} is broken",
                        f"someone should tell {bot.NICK} that it is wrong"):
            with self.subTest(message=message):
                self.assertIsNone(bot._match_trigger(message))

    def test_a_bare_greeting_is_not_a_prompt(self):
        self.assertIsNone(bot._match_trigger(f"hey {bot.NICK}"))

    def test_a_greeting_without_the_nick_is_ignored(self):
        self.assertIsNone(bot._match_trigger("hey everyone, whats up"))


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


class TestDirectiveModes(unittest.TestCase):
    """science / research / answer ask for a serious, concise answer -- NOT a
    fact-check verdict. Tested against llmbot_core (the active module); the
    frozen bot.py keeps the old science:/research: -> factual behaviour."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._pending["prompt"] = ""
        llmbot_core._end_conversation()

    NICK = llmbot_core.NICK

    def test_research_without_colon(self):
        self.assertEqual(
            llmbot_core._match_trigger("Research dangers of lead"),
            (llmbot_core.MODE_RESEARCH, "dangers of lead"))

    def test_answer_with_filler_before_and_after(self):
        self.assertEqual(
            llmbot_core._match_trigger(f"{self.NICK} can you answer this or that"),
            (llmbot_core.MODE_ANSWER, "this or that"))

    def test_research_with_lead_in_and_nick(self):
        self.assertEqual(
            llmbot_core._match_trigger(f"hey {self.NICK}, research this and that"),
            (llmbot_core.MODE_RESEARCH, "this and that"))

    def test_science_with_colon_is_no_verdict_mode(self):
        self.assertEqual(
            llmbot_core._match_trigger(f"{self.NICK}: science why is the sky blue"),
            (llmbot_core.MODE_SCIENCE, "why is the sky blue"))

    def test_factcheck_still_selects_factual(self):
        self.assertEqual(
            llmbot_core._match_trigger("factcheck whales are fish"),
            (llmbot_core.MODE_FACTUAL, "whales are fish"))

    def test_directive_followed_by_verb_is_not_a_directive(self):
        """'research shows ...' / 'science is ...' are statements, not calls."""
        self.assertIsNone(
            llmbot_core._match_trigger("research shows that irc is dead"))
        self.assertIsNone(
            llmbot_core._match_trigger("science is great and you know it"))

    def test_command_word_as_noun_is_not_a_directive(self):
        self.assertIsNone(
            llmbot_core._match_trigger("the answer to life is 42"))
        self.assertIsNone(
            llmbot_core._match_trigger("a science experiment is controlled"))

    def test_command_word_without_addressing_is_not_a_directive(self):
        """'I need to research this' is someone else's plan, not a call."""
        self.assertIsNone(
            llmbot_core._match_trigger("I need to research this for school"))
        self.assertIsNone(llmbot_core._match_trigger("can you research this"))

    def test_science_research_answer_share_one_persona(self):
        self.assertEqual(
            llmbot_core._system_prompt(llmbot_core.MODE_SCIENCE),
            llmbot_core._system_prompt(llmbot_core.MODE_RESEARCH))
        self.assertEqual(
            llmbot_core._system_prompt(llmbot_core.MODE_RESEARCH),
            llmbot_core._system_prompt(llmbot_core.MODE_ANSWER))

    def test_directive_persona_has_no_verdict_instruction(self):
        """The fact-checker opens with a verdict word; the directive modes must
        not -- they just answer the question."""
        factual = llmbot_core._system_prompt(llmbot_core.MODE_FACTUAL)
        science = llmbot_core._system_prompt(llmbot_core.MODE_SCIENCE)
        self.assertIn("start your reply with TRUE", factual)
        self.assertNotIn("start your reply with TRUE", science)
        self.assertIn("just answer the question", science)

    def test_directive_modes_are_context_free(self):
        """Like factual, the directive modes answer about the world, not the
        people in the room, so the userlist is not woven into their prompt."""
        for mode in (llmbot_core.MODE_SCIENCE, llmbot_core.MODE_RESEARCH,
                     llmbot_core.MODE_ANSWER):
            with self.subTest(mode=mode):
                self.assertEqual(
                    llmbot_core._system_context(mode),
                    llmbot_core._system_prompt(mode))

    def test_directive_mode_survives_the_pending_queue(self):
        """The directive mode captured by the receiver must reach the LLM call."""
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "Lead poisons the nervous system."
        llmbot_core._handle_ai_prompt(sock, "alice", "Research dangers of lead")
        with mock.patch.object(llmbot_core._llm_client.chat.completions, "create", return_value=mock_response) as create:
            llmbot_core._process_pending(sock)
        self.assertEqual(
            create.call_args.kwargs["messages"][0]["content"],
            llmbot_core._system_prompt(llmbot_core.MODE_RESEARCH))


class TestFollowUpWindowLength(unittest.TestCase):
    def test_window_is_a_sane_length(self):
        """A hand-tuned knob: assert it is plausible, not one exact value."""
        self.assertGreaterEqual(bot.FOLLOWUP_WINDOW, 5.0)
        self.assertLessEqual(bot.FOLLOWUP_WINDOW, 120.0)

    def test_window_is_shorter_than_the_silence_timeout(self):
        self.assertLess(bot.FOLLOWUP_WINDOW, bot.SILENCE_TIMEOUT)


class TestUnpromptedInterjection(unittest.TestCase):
    """After enough unaddressed chatter, the bot chimes in on its own."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
            bot._pending["stop"] = False
        bot._end_conversation()
        bot._reset_chatter()
        bot._joined["at"] = 0.0
        bot._set_mood(bot.MOOD_BANTER)

    def _chatter(self, count, text="just people talking"):
        sock = mock.MagicMock(spec=socket.socket)
        for i in range(count):
            bot._handle_ai_prompt(sock, "alice", f"{text} {i}")
            bot._note_chatter(f"{text} {i}")

    def test_stays_quiet_below_the_threshold(self):
        self._chatter(bot.IDLE_INTERJECT_AFTER - 1)
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_interjects_at_the_threshold(self):
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        self.assertNotEqual(bot.get_pending_prompt(), "")

    def test_no_interject_during_join_grace(self):
        bot._joined["at"] = time.monotonic() - (bot.JOIN_GRACE_PERIOD - 0.1)
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_interject_allowed_after_join_grace(self):
        bot._joined["at"] = time.monotonic() - (bot.JOIN_GRACE_PERIOD + 5)
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        self.assertNotEqual(bot.get_pending_prompt(), "")

    def test_low_roll_replies_to_the_last_message(self):
        with mock.patch.object(bot.random, "random", return_value=0.1):
            self._chatter(bot.IDLE_INTERJECT_AFTER, text="my server caught fire")
        self.assertEqual(
            bot.get_pending_prompt(),
            f"my server caught fire {bot.IDLE_INTERJECT_AFTER - 1}")

    def test_high_roll_asks_for_something_funny(self):
        with mock.patch.object(bot.random, "random", return_value=0.9):
            self._chatter(bot.IDLE_INTERJECT_AFTER)
        self.assertEqual(bot.get_pending_prompt(), bot.IDLE_PROMPT)

    def test_split_is_even(self):
        """Both branches must be reachable across the 0..1 range."""
        seen = set()
        for roll in (0.0, 0.49, 0.5, 0.99):
            bot._reset_chatter()
            with mock.patch.object(bot.random, "random", return_value=roll):
                self._chatter(bot.IDLE_INTERJECT_AFTER)
            seen.add(bot.get_pending_prompt() == bot.IDLE_PROMPT)
        self.assertEqual(seen, {True, False})

    def test_counter_resets_after_interjecting(self):
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        bot.get_pending_prompt()
        self._chatter(bot.IDLE_INTERJECT_AFTER - 1)
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_being_addressed_resets_the_counter(self):
        sock = mock.MagicMock(spec=socket.socket)
        self._chatter(bot.IDLE_INTERJECT_AFTER - 1)
        bot._handle_ai_prompt(sock, "bob", f"{bot.NICK}: hello")
        bot.get_pending_prompt()
        bot._end_conversation()
        self._chatter(1)
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_interjection_uses_interject_mode(self):
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        with bot._prompt_lock:
            self.assertEqual(bot._pending["mode"], bot.MODE_INTERJECT)

    def test_interject_prompt_keeps_the_channel_persona(self):
        """Unprompted lines are still Heretic, just steered to banter."""
        self.assertIn(bot._system_prompt(bot.MODE_CHAT),
                      bot._system_prompt(bot.MODE_INTERJECT))

    def test_interjection_does_not_open_a_follow_up_window(self):
        """Nobody addressed the bot, so it must not latch onto them."""
        self._chatter(bot.IDLE_INTERJECT_AFTER)
        self.assertFalse(bot._in_conversation_with("alice"))

    def test_interjection_does_not_overwrite_a_real_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        self._chatter(bot.IDLE_INTERJECT_AFTER - 1)
        bot._handle_ai_prompt(sock, "bob", f"{bot.NICK}: answer me")
        bot._note_chatter("more chatter")
        self.assertEqual(bot.get_pending_prompt(), "answer me")


class TestSilenceBreaker(unittest.TestCase):
    """After a long silence the bot speaks up, then opens the floor briefly."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
            bot._pending["stop"] = False
        bot._end_conversation()
        bot._reset_chatter()
        bot._close_open_floor()
        bot._note_activity()
        bot._joined["at"] = 0.0
        bot._set_mood(bot.MOOD_BANTER)

    def _go_quiet(self, seconds=None):
        """Pretend the channel has been silent for `seconds`."""
        seconds = bot.SILENCE_TIMEOUT if seconds is None else seconds
        with bot._prompt_lock:
            bot._activity["at"] = time.monotonic() - seconds

    def test_constants(self):
        self.assertEqual(bot.SILENCE_TIMEOUT, 30 * 60)
        self.assertEqual(bot.OPEN_FLOOR_WINDOW, 60.0)
        self.assertEqual(bot.OPEN_FLOOR_MAX_PROMPTS, 8)

    def test_quiet_channel_below_the_timeout_stays_quiet(self):
        self._go_quiet(bot.SILENCE_TIMEOUT - 60)
        self.assertFalse(bot._check_silence())
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_breaks_a_long_silence(self):
        self._go_quiet()
        self.assertTrue(bot._check_silence())
        self.assertNotEqual(bot.get_pending_prompt(), "")

    def test_interject_deferred_until_after_join_grace(self):
        # Joined moments ago: defer the opener so the userlist has time to arrive.
        with bot._prompt_lock:
            bot._joined["at"] = time.monotonic()
        self._go_quiet(bot.SILENCE_TIMEOUT + 60)
        self.assertFalse(bot._check_silence())
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_interject_fires_after_join_grace(self):
        with bot._prompt_lock:
            bot._joined["at"] = time.monotonic() - (bot.JOIN_GRACE_PERIOD + 1)
        self._go_quiet(bot.SILENCE_TIMEOUT + 60)
        self.assertTrue(bot._check_silence())

    def test_silence_breaker_uses_interject_mode(self):
        self._go_quiet()
        bot._check_silence()
        with bot._prompt_lock:
            self.assertEqual(bot._pending["mode"], bot.MODE_INTERJECT)

    def test_does_not_fire_twice_for_one_silence(self):
        self._go_quiet()
        bot._check_silence()
        bot.get_pending_prompt()
        self.assertFalse(bot._check_silence())

    def test_does_not_fire_over_a_waiting_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: answer me")
        self._go_quiet()
        self.assertFalse(bot._check_silence())
        self.assertEqual(bot.get_pending_prompt(), "answer me")

    def test_any_message_resets_the_silence_timer(self):
        sock = mock.MagicMock(spec=socket.socket)
        self._go_quiet()
        bot._handle_ai_prompt(sock, "alice", "just chatting")
        self.assertFalse(bot._check_silence())


class TestOpenFloor(unittest.TestCase):
    """The minute after a silence breaker, anyone can talk to the bot."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
            bot._pending["stop"] = False
        bot._end_conversation()
        bot._reset_chatter()
        bot._close_open_floor()
        bot._note_activity()
        bot._joined["at"] = 0.0
        bot._set_mood(bot.MOOD_BANTER)
        with bot._prompt_lock:
            bot._activity["at"] = time.monotonic() - bot.SILENCE_TIMEOUT
        bot._check_silence()
        bot.get_pending_prompt()

    def test_untriggered_message_is_answered(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "oh youre awake")
        self.assertEqual(bot.get_pending_prompt(), "oh youre awake")

    def test_any_user_not_just_one(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "hello")
        bot.get_pending_prompt()
        bot._handle_ai_prompt(sock, "bob", "hello too")
        self.assertEqual(bot.get_pending_prompt(), "hello too")

    def test_caps_at_the_prompt_limit(self):
        sock = mock.MagicMock(spec=socket.socket)
        for i in range(bot.OPEN_FLOOR_MAX_PROMPTS):
            bot._handle_ai_prompt(sock, "alice", f"line {i}")
            self.assertEqual(bot.get_pending_prompt(), f"line {i}")
        bot._handle_ai_prompt(sock, "alice", "one too many")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_direct_address_still_works_after_the_cap(self):
        sock = mock.MagicMock(spec=socket.socket)
        for i in range(bot.OPEN_FLOOR_MAX_PROMPTS):
            bot._handle_ai_prompt(sock, "alice", f"line {i}")
            bot.get_pending_prompt()
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: oi")
        self.assertEqual(bot.get_pending_prompt(), "oi")

    def test_closes_when_the_window_expires(self):
        sock = mock.MagicMock(spec=socket.socket)
        with bot._prompt_lock:
            bot._open_floor["deadline"] = time.monotonic() - 1
        bot._handle_ai_prompt(sock, "alice", "too late")
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_shut_up_closes_the_floor(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "shut up")
        with mock.patch.object(bot._llm_client.chat.completions, "create"):
            bot._process_pending(sock)
        bot._handle_ai_prompt(sock, "bob", "anything")
        self.assertEqual(bot.get_pending_prompt(), "")


class TestSystemPrompt(unittest.TestCase):
    """The persona must follow the bot's identity and stay uncensored."""

    def setUp(self):
        with bot._prompt_lock:
            bot._users["names"].clear()
            bot._recent_lines.clear()

    def test_temperature_is_sent_explicitly(self):
        """The bot pins its own temperature; the server's --temp is retuned for
        other models and must not leak into the channel's persona."""
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hello")
        self.assertEqual(create.call_args.kwargs["temperature"], bot.LLM_TEMPERATURE)

    def test_temperature_is_a_valid_sampling_value(self):
        """Temperature is pinned for this bot; guard that the pin is sane rather
        than hard-coding the current value, which may change with tuning."""
        self.assertIsInstance(bot.LLM_TEMPERATURE, (int, float))
        self.assertGreaterEqual(bot.LLM_TEMPERATURE, 0.0)
        self.assertLessEqual(bot.LLM_TEMPERATURE, 2.0)

    def test_is_sent_with_every_request(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hello")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], bot._system_context(bot.MODE_CHAT))


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


class TestMood(unittest.TestCase):
    """banter / serious are global moods that outlive a single reply."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
            bot._pending["stop"] = False
        bot._end_conversation()
        bot._close_open_floor()
        bot._reset_chatter()
        bot._set_mood(bot.MOOD_BANTER)

    def tearDown(self):
        bot._set_mood(bot.MOOD_BANTER)

    def _age_the_mood(self, seconds):
        """Pretend the current mood was set `seconds` ago."""
        with bot._prompt_lock:
            bot._mood["at"] = time.monotonic() - seconds

    def test_mood_timeout_is_fifteen_minutes(self):
        self.assertEqual(bot.MOOD_TIMEOUT, 15 * 60)

    def test_every_mood_announces_itself(self):
        for mood in (bot.MOOD_BANTER, bot.MOOD_SERIOUS, bot.MOOD_FACTUAL):
            with self.subTest(mood=mood):
                bot._set_mood(bot.MOOD_BANTER if mood != bot.MOOD_BANTER
                              else bot.MOOD_SERIOUS)
                sock = mock.MagicMock(spec=socket.socket)
                bot._handle_ai_prompt(sock, "alice", mood)
                sends = [call.args[0] for call in sock.send.call_args_list]
                self.assertEqual(
                    sends,
                    [f"PRIVMSG {bot.CHANNEL} :{bot.MOOD_REPLIES[mood]}\r\n".encode()])

    def test_the_announcements_are_the_channels_own_words(self):
        self.assertEqual(bot.MOOD_REPLIES[bot.MOOD_SERIOUS],
                         "Ok I'll be serious for a while")
        self.assertEqual(bot.MOOD_REPLIES[bot.MOOD_BANTER],
                         "Oh you want bants huh? Fine")
        self.assertEqual(bot.MOOD_REPLIES[bot.MOOD_FACTUAL],
                         "Factchecking engaged")

    def test_startup_mood_is_a_coin_flip_between_the_two(self):
        with mock.patch.object(
            bot.random, "choice", return_value=bot.MOOD_SERIOUS
        ) as choice:
            self.assertEqual(bot._random_mood(), bot.MOOD_SERIOUS)
        self.assertEqual(set(choice.call_args.args[0]),
                         {bot.MOOD_BANTER, bot.MOOD_SERIOUS})

    def test_bare_word_switches_the_mood(self):
        for word, mood in (("serious", bot.MOOD_SERIOUS),
                           ("banter", bot.MOOD_BANTER)):
            with self.subTest(word=word):
                sock = mock.MagicMock(spec=socket.socket)
                bot._handle_ai_prompt(sock, "alice", word)
                self.assertEqual(bot._current_mood(), mood)

    def test_addressed_forms_switch_the_mood(self):
        for message in (f"{bot.NICK}: serious", f"{bot.NICK}, serious",
                        "AI: serious", "SERIOUS", "serious!", "  serious  "):
            with self.subTest(message=message):
                bot._set_mood(bot.MOOD_BANTER)
                sock = mock.MagicMock(spec=socket.socket)
                bot._handle_ai_prompt(sock, "alice", message)
                self.assertEqual(bot._current_mood(), bot.MOOD_SERIOUS)

    def test_padded_command_switches_when_addressed(self):
        """"be serious" is an order when it is aimed at the bot."""
        for message in (f"{bot.NICK}, be serious",
                        f"{bot.NICK}: be serious for once",
                        f"hey {bot.NICK}, be a bit more serious please",
                        f"{bot.NICK}: serious mode",
                        "AI: get serious now",
                        f"be serious, {bot.NICK}"):
            with self.subTest(message=message):
                bot._set_mood(bot.MOOD_BANTER)
                sock = mock.MagicMock(spec=socket.socket)
                bot._handle_ai_prompt(sock, "alice", message)
                self.assertEqual(bot._current_mood(), bot.MOOD_SERIOUS)

    def test_padded_command_switches_back_to_banter(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: banter mode please")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_padded_command_needs_the_bot_to_be_addressed(self):
        """"be serious" between two humans is not the bot's business."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "bob be serious for once")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_padded_command_works_mid_conversation(self):
        bot._note_conversation("alice")
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "be serious")
        self.assertEqual(bot._current_mood(), bot.MOOD_SERIOUS)

    def test_the_word_inside_a_sentence_is_not_a_command(self):
        for message in ("are you serious", "seriously though", "banter is fun",
                        "serious question, how tall is everest",
                        f"{bot.NICK}: are you serious",
                        f"{bot.NICK}: is it serious",
                        f"{bot.NICK}: why so serious",
                        f"{bot.NICK}: stop being serious",
                        f"{bot.NICK}: how serious is that bug"):
            with self.subTest(message=message):
                self.assertIsNone(bot._match_mood_command("alice", message))

    def test_a_non_command_is_still_answered_as_a_prompt(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: are you serious")
        self.assertEqual(bot.get_pending_prompt(), "are you serious")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_command_is_acknowledged_without_an_llm_call(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "serious")
        sends = [call.args[0] for call in sock.send.call_args_list]
        self.assertTrue(any(b"PRIVMSG #hive :" in s for s in sends))
        self.assertEqual(bot.get_pending_prompt(), "")

    def test_command_counts_as_addressing_the_bot(self):
        """So the receiver does not also file it as unaddressed chatter."""
        sock = mock.MagicMock(spec=socket.socket)
        self.assertTrue(bot._handle_ai_prompt(sock, "alice", "banter"))

    def test_serious_sticks_across_replies(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        for _ in range(5):
            self.assertEqual(bot._effective_mode(bot.MODE_CHAT), bot.MODE_SERIOUS)

    def test_banter_command_ends_serious_before_the_timer(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "serious")
        bot._handle_ai_prompt(sock, "bob", "banter")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_serious_survives_up_to_the_timeout(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        self._age_the_mood(bot.MOOD_TIMEOUT - 60)
        self.assertEqual(bot._current_mood(), bot.MOOD_SERIOUS)

    def test_serious_lapses_back_to_banter(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        self._age_the_mood(bot.MOOD_TIMEOUT)
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)
        self.assertEqual(bot._effective_mode(bot.MODE_CHAT), bot.MODE_CHAT)

    def test_repeating_the_command_restarts_the_timer(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        self._age_the_mood(bot.MOOD_TIMEOUT - 60)
        with bot._prompt_lock:
            before = bot._mood["at"]
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "serious")
        with bot._prompt_lock:
            self.assertGreater(bot._mood["at"], before)

    def test_banter_never_expires(self):
        bot._set_mood(bot.MOOD_BANTER)
        self._age_the_mood(bot.MOOD_TIMEOUT * 10)
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_bare_factcheck_switches_the_mood(self):
        for message in ("factcheck", "factchecking", f"{bot.NICK}: factcheck",
                        f"hey {bot.NICK}, factchecking mode please",
                        "AI: factcheck"):
            with self.subTest(message=message):
                bot._set_mood(bot.MOOD_BANTER)
                bot.get_pending_prompt()
                sock = mock.MagicMock(spec=socket.socket)
                bot._handle_ai_prompt(sock, "alice", message)
                self.assertEqual(bot._current_mood(), bot.MOOD_FACTUAL)
                self.assertEqual(bot.get_pending_prompt(), "")

    def test_a_factcheck_with_a_claim_is_still_a_one_off(self):
        """"factcheck X" answers X; it must not put the channel in the mood."""
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "factcheck whales are fish")
        self.assertEqual(bot.get_pending_prompt(), "whales are fish")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_factual_mood_answers_chat_in_the_factual_persona(self):
        bot._set_mood(bot.MOOD_FACTUAL)
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "TRUE."
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: is the sky blue")
        with mock.patch.object(
            bot._llm_client.chat.completions, "create", return_value=mock_response
        ) as create:
            bot._process_pending(sock)
        self.assertEqual(create.call_args.kwargs["messages"][0]["content"],
                         bot._system_prompt(bot.MODE_FACTUAL))

    def test_factual_mood_lapses_back_to_banter(self):
        bot._set_mood(bot.MOOD_FACTUAL)
        self._age_the_mood(bot.MOOD_TIMEOUT)
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_banter_ends_the_factual_mood(self):
        bot._set_mood(bot.MOOD_FACTUAL)
        sock = mock.MagicMock(spec=socket.socket)
        bot._handle_ai_prompt(sock, "alice", "banter")
        self.assertEqual(bot._current_mood(), bot.MOOD_BANTER)

    def test_boot_mood_is_never_factchecking(self):
        """Booting as a fact-checker nobody asked for is the worst surprise."""
        drawn = {bot._random_mood() for _ in range(50)}
        self.assertEqual(drawn - {bot.MOOD_BANTER, bot.MOOD_SERIOUS}, set())

    def test_serious_mood_replaces_both_banter_personas(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        self.assertEqual(bot._effective_mode(bot.MODE_CHAT), bot.MODE_SERIOUS)
        self.assertEqual(bot._effective_mode(bot.MODE_INTERJECT), bot.MODE_SERIOUS)

    def test_banter_mood_leaves_the_modes_alone(self):
        self.assertEqual(bot._effective_mode(bot.MODE_CHAT), bot.MODE_CHAT)
        self.assertEqual(bot._effective_mode(bot.MODE_INTERJECT),
                         bot.MODE_INTERJECT)

    def test_factcheck_is_untouched_by_the_mood(self):
        """An explicit factcheck asked for the factual prompt by name."""
        for mood in (bot.MOOD_BANTER, bot.MOOD_SERIOUS, bot.MOOD_FACTUAL):
            with self.subTest(mood=mood):
                bot._set_mood(mood)
                self.assertEqual(bot._effective_mode(bot.MODE_FACTUAL),
                                 bot.MODE_FACTUAL)

    def test_serious_mood_reaches_the_llm_call(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "42."
        bot._handle_ai_prompt(sock, "alice", f"{bot.NICK}: what is 6 times 7")
        with mock.patch.object(
            bot._llm_client.chat.completions, "create", return_value=mock_response
        ) as create:
            bot._process_pending(sock)
        self.assertEqual(create.call_args.kwargs["messages"][0]["content"],
                         bot._system_context(bot.MODE_SERIOUS))

    def test_serious_persona_is_neither_the_chat_nor_the_factual_prompt(self):
        self.assertNotEqual(bot._system_prompt(bot.MODE_SERIOUS),
                            bot._system_prompt(bot.MODE_CHAT))
        self.assertNotEqual(bot._system_prompt(bot.MODE_SERIOUS),
                            bot._system_prompt(bot.MODE_FACTUAL))

    def test_serious_interjection_does_not_ask_for_a_joke(self):
        bot._set_mood(bot.MOOD_SERIOUS)
        with mock.patch.object(bot.random, "random", return_value=0.9):
            bot._queue_interjection("")
        self.assertEqual(bot.get_pending_prompt(), bot.SERIOUS_IDLE_PROMPT)


class TestWhoRequest(unittest.TestCase):
    """The bot asks the server who is in the channel, after joining."""

    def test_who_sent_after_join(self):
        sock = mock.MagicMock(spec=socket.socket)
        bot._request_userlist(sock)
        sock.send.assert_called_once_with(b"WHO #hive\r\n")


class TestUserListParsing(unittest.TestCase):
    """Parse the IRC userlist replies into nicks."""

    def test_who_reply_returns_nick_after_channel(self):
        line = (
            ":hive.2bd.net 352 Heretic #hive alice a.host.hive.2bd.net "
            "hive.2bd.net alice (H) 0 :Alice Example"
        )
        self.assertEqual(bot._parse_who_reply(line), "alice")

    def test_name_reply_strips_prefixes(self):
        line = ":hive.2bd.net 353 Heretic #hive :@alice +bob carol"
        self.assertEqual(bot._parse_name_reply(line), ["alice", "bob", "carol"])

    def test_strips_all_status_prefixes(self):
        line = (
            ":hive.2bd.net 353 Heretic #hive :@ops +voice &admin %halfnick carol"
        )
        self.assertEqual(
            bot._parse_name_reply(line),
            ["ops", "voice", "admin", "halfnick", "carol"],
        )

    def test_who_reply_strips_status_prefix(self):
        line = (
            ":hive.2bd.net 352 Heretic #hive alice a.host.hive.2bd.net "
            "hive.2bd.net @alice (H) 0 :Alice Example"
        )
        self.assertEqual(bot._parse_who_reply(line), "alice")

    def test_who_reply_none_without_channel(self):
        self.assertIsNone(bot._parse_who_reply(":server 352 Heretic"))


class TestUserRegistration(unittest.TestCase):
    """The channel members are remembered, minus the bot itself."""

    def setUp(self):
        with bot._prompt_lock:
            bot._users["names"].clear()

    def test_member_is_registered(self):
        bot._register_user("alice")
        self.assertIn("alice", bot._channel_users())

    def test_own_nick_is_not_registered(self):
        bot._register_user(bot.NICK)
        self.assertNotIn(bot.NICK, bot._channel_users())

    def test_members_are_deduplicated(self):
        bot._register_user("alice")
        bot._register_user("alice")
        self.assertEqual(bot._channel_users(), ["alice"])


class TestSystemContext(unittest.TestCase):
    """The userlist is woven into the chat and interjection personas only."""

    def setUp(self):
        with bot._prompt_lock:
            bot._users["names"].clear()
            bot._recent_lines.clear()
            bot._recent_senders.clear()
            bot._conversation["nick"] = ""

    def _set(self, names):
        with bot._prompt_lock:
            bot._users["names"] = list(names)

    def test_chat_mode_includes_user_list(self):
        self._set(["alice", "bob"])
        ctx = bot._system_context(bot.MODE_CHAT)
        self.assertIn(
            "The users in this IRC channel are named: alice, bob", ctx
        )

    def test_interject_mode_includes_user_list(self):
        self._set(["alice"])
        ctx = bot._system_context(bot.MODE_INTERJECT)
        self.assertIn(
            "The users in this IRC channel are named: alice", ctx
        )

    def test_factual_mode_excludes_user_list(self):
        self._set(["alice", "bob"])
        ctx = bot._system_context(bot.MODE_FACTUAL)
        self.assertNotIn("The users in this IRC channel are named:", ctx)

    def test_silent_when_no_users(self):
        ctx = bot._system_context(bot.MODE_CHAT)
        self.assertNotIn("The users in this IRC channel are named:", ctx)

    def test_recent_lines_sent_as_messages(self):
        # Recent history goes into the LLM call as messages -- named by sender,
        # between the system prompt and the user's message -- not pasted into
        # the system prompt.
        self._set(["alice"])
        with bot._prompt_lock:
            bot._recent_lines.extend(["hello there", "how's it going"])
            bot._recent_senders.extend(["alice", "bob"])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hey")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "alice: hello there")
        self.assertNotIn("name", messages[1])
        self.assertEqual(messages[2]["content"], "bob: how's it going")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], "hey")

    def test_recent_lines_have_no_name_field(self):
        # Sender is encoded inline in content, not a separate "name" field:
        # the field is OpenAI-specific and open models parse "alice: hi" better.
        with bot._prompt_lock:
            bot._recent_lines.extend(["hi"])
            bot._recent_senders.extend(["alice"])
        msg = bot._recent_messages()[0]
        self.assertEqual(msg, {"role": "user", "content": "alice: hi"})

    def test_recent_lines_sent_regardless_of_mode(self):
        # Injected on every mode -- factual included -- because a reply is
        # always inside an ongoing room.
        self._set(["alice"])
        with bot._prompt_lock:
            bot._recent_lines.extend(["hello there"])
            bot._recent_senders.extend(["alice"])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hey", bot.MODE_FACTUAL)
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 3)  # system + one recent line + user
        self.assertEqual(messages[1]["content"], "alice: hello there")

    def test_recent_lines_capped_at_100(self):
        with bot._prompt_lock:
            bot._recent_lines.extend(str(i) for i in range(200))
            bot._recent_senders.extend(["alice"] * 200)
        messages = bot._recent_messages()
        self.assertEqual(len(messages), 100)
        self.assertEqual(messages[0]["content"], "alice: 100")

    def test_recent_lines_logged_at_call(self):
        with bot._prompt_lock:
            bot._recent_lines.extend(["hello there", "how's it going"])
            bot._recent_senders.extend(["alice", "bob"])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response):
            with mock.patch("builtins.print", wrap=print) as p:
                bot._call_llm("hey")
        p.assert_any_call("Injected 2 lines of chat history as context", flush=True)

    def test_system_prompt_has_no_recent_lines(self):
        # Recent history moved out of the system prompt into the messages list.
        self._set(["alice"])
        with bot._prompt_lock:
            bot._recent_lines.extend(["hello there"])
        self.assertNotIn("Recent channel messages:", bot._system_context(bot.MODE_CHAT))

    def test_serious_mode_includes_context(self):
        # Only factual is context-free; serious still gets the userlist in the
        # prompt and the recent history in the messages.
        self._set(["alice"])
        with bot._prompt_lock:
            bot._recent_lines.extend(["hello there"])
            bot._recent_senders.extend(["alice"])
        self.assertIn(
            "The users in this IRC channel are named: alice",
            bot._system_context(bot.MODE_SERIOUS),
        )
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "hi"
        with mock.patch.object(bot._llm_client.chat.completions, "create", return_value=mock_response) as create:
            bot._call_llm("hey", bot.MODE_SERIOUS)
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[1]["content"], "alice: hello there")


class TestMentionTargets(unittest.TestCase):
    """The mention list favours who engaged the bot / spoke recently, not random."""

    def setUp(self):
        with bot._prompt_lock:
            bot._users["names"].clear()
            bot._recent_lines.clear()
            bot._recent_senders.clear()
            bot._conversation["nick"] = ""

    def _users(self, *names):
        with bot._prompt_lock:
            bot._users["names"] = list(names)

    def _recent(self, *senders):
        with bot._prompt_lock:
            bot._recent_senders.clear()
            for nick in senders:
                bot._recent_senders.append(nick)

    def test_falls_back_to_registration_order_with_no_activity(self):
        self._users("alice", "bob", "carol")
        self.assertEqual(bot._mention_targets(), ["alice", "bob", "carol"])

    def test_prefers_the_addressed_person(self):
        self._users("alice", "bob", "carol")
        with bot._prompt_lock:
            bot._conversation["nick"] = "bob"
        self.assertEqual(bot._mention_targets()[0], "bob")

    def test_falls_back_to_last_speaker_when_no_one_addressed(self):
        self._users("alice", "bob", "carol")
        self._recent("alice", "bob", "carol")
        self.assertEqual(bot._mention_targets()[0], "carol")

    def test_recent_speakers_come_before_others(self):
        self._users("alice", "bob", "carol", "dave")
        self._recent("alice", "dave")
        targets = bot._mention_targets()
        # Both recent speakers (dave, alice) precede the idle members (bob,
        # carol); dave is most recent so comes before alice.
        self.assertEqual(targets, ["dave", "alice", "bob", "carol"])

    def test_dedupes_across_tiers(self):
        self._users("alice", "bob")
        with bot._prompt_lock:
            bot._conversation["nick"] = "alice"
        self._recent("alice", "bob")
        targets = bot._mention_targets()
        self.assertEqual(set(targets), {"alice", "bob"})
        self.assertEqual(targets.count("alice"), 1)

    def test_addressed_but_absent_falls_back_to_last_speaker(self):
        self._users("alice", "bob")
        with bot._prompt_lock:
            bot._conversation["nick"] = "ghost"
        self._recent("alice", "bob")
        self.assertEqual(bot._mention_targets()[0], "bob")

    def test_no_recent_lines_fills_tier_with_other_members(self):
        # Fresh join: no last-15 lines yet, so the recent-speak tier is empty
        # and those slots fall through to the other channel members.
        self._users("alice", "bob")
        self.assertEqual(bot._mention_targets(), ["alice", "bob"])

    def test_excludes_its_own_nick(self):
        self._users(bot.NICK, "alice")
        self.assertEqual(bot._mention_targets(), ["alice"])

    def test_context_orders_by_priority_and_states_the_preference(self):
        self._users("alice", "bob", "carol")
        with bot._prompt_lock:
            bot._conversation["nick"] = "carol"
        ctx = bot._system_context(bot.MODE_CHAT)
        # carol (who addressed the bot) is named first, then the rest.
        self.assertIn(
            "The users in this IRC channel are named: carol, alice, bob", ctx
        )
        self.assertIn("Prefer to mention the first one", ctx)


class TestStatusSnapshotUsersOrder(unittest.TestCase):
    """The chatter list shown in the TUI must be ordered by recency, matching
    the order the names are passed to the LLM (_mention_targets).

    The status pane is rendered from llmbot_core.status_snapshot(), so this
    suite exercises that real output rather than bot.py (which has no snapshot).
    """

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._users["names"] = []
            llmbot_core._recent_senders.clear()
            llmbot_core._conversation["nick"] = ""

    def tearDown(self):
        with llmbot_core._prompt_lock:
            llmbot_core._users["names"] = []
            llmbot_core._recent_senders.clear()
            llmbot_core._conversation["nick"] = ""

    def _users(self, *names):
        with llmbot_core._prompt_lock:
            llmbot_core._users["names"] = list(names)

    def _recent(self, *senders):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_senders.clear()
            for nick in senders:
                llmbot_core._recent_senders.append(nick)

    def test_displayed_users_match_the_llm_mention_order(self):
        self._users("alice", "bob", "carol", "dave")
        self._recent("alice", "dave")
        self.assertEqual(
            llmbot_core.status_snapshot()["users"], llmbot_core._mention_targets()
        )

    def test_displayed_users_sorted_most_recent_first(self):
        self._users("alice", "bob", "carol", "dave")
        self._recent("alice", "dave")
        # dave (most recent) and alice precede the idle members bob, carol.
        self.assertEqual(
            llmbot_core.status_snapshot()["users"], ["dave", "alice", "bob", "carol"]
        )

    def test_displayed_users_include_non_speakers(self):
        self._users("alice", "bob", "carol")
        self.assertEqual(
            set(llmbot_core.status_snapshot()["users"]), {"alice", "bob", "carol"}
        )

    def test_displayed_users_exclude_its_own_nick(self):
        self._users(llmbot_core.NICK, "alice")
        self.assertNotIn(
            llmbot_core.NICK, llmbot_core.status_snapshot()["users"]
        )


class TestCoreSelfFiltering(unittest.TestCase):
    """llmbot_core (the TUI fork) must not treat the bot itself as a chatter.

    bot.py has its own copies of this logic; this suite exercises the module
    the TUI actually runs. Three things must hold: the chatter list excludes
    the companion nick 'Botmans'; the LLM history excludes the bot's own
    echoed messages (so it never talks about 'sloppy' in the 3rd person); and
    the system prompt tells the model to refer to itself in the first person.
    """

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._users["names"] = []
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()
            llmbot_core._conversation["nick"] = ""

    def tearDown(self):
        with llmbot_core._prompt_lock:
            llmbot_core._users["names"] = []
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()
            llmbot_core._conversation["nick"] = ""

    def test_register_user_excludes_companion_nick(self):
        llmbot_core._register_user("Botmans")
        self.assertNotIn("Botmans", llmbot_core._channel_users())
        self.assertNotIn("Botmans", llmbot_core._mention_targets())

    def test_register_user_still_registers_others(self):
        llmbot_core._register_user("alice")
        llmbot_core._register_user("bob")
        self.assertEqual(llmbot_core._channel_users(), ["alice", "bob"])

    def test_recent_history_excludes_its_own_messages(self):
        llmbot_core._note_recent("hey what are we doing", llmbot_core.NICK)
        self.assertEqual(list(llmbot_core._recent_senders), [])
        self.assertEqual(llmbot_core._recent_messages(), [])

    def test_recent_history_excludes_own_nick_any_case(self):
        # IRC nicks are case-insensitive; the server may echo a different case.
        llmbot_core._note_recent("echo", llmbot_core.NICK.upper())
        self.assertEqual(list(llmbot_core._recent_senders), [])

    def test_recent_history_records_others_while_skipping_own(self):
        llmbot_core._note_recent("first message", "alice")
        llmbot_core._note_recent("my own message", llmbot_core.NICK)
        llmbot_core._note_recent("third message", "bob")
        self.assertEqual(list(llmbot_core._recent_senders), ["alice", "bob"])
        contents = [m["content"] for m in llmbot_core._recent_messages()]
        self.assertNotIn(f"{llmbot_core.NICK}: my own message", contents)
        self.assertIn("alice: first message", contents)
        self.assertIn("bob: third message", contents)

    def test_system_prompt_tells_model_to_use_first_person(self):
        ctx = llmbot_core._system_prompt(llmbot_core.MODE_CHAT)
        self.assertIn("Refer to yourself as I or me", ctx)
        self.assertIn(f"you ARE {llmbot_core.NICK}", ctx)

    def test_interject_prompt_carries_first_person_rule(self):
        # INTERJECT layers on the chat persona, so the rule must survive.
        ctx = llmbot_core._system_prompt(llmbot_core.MODE_INTERJECT)
        self.assertIn("Refer to yourself as I or me", ctx)


class TestReceiverUserlist(unittest.TestCase):
    """The receiver records members from the channel userlist."""

    def setUp(self):
        with bot._prompt_lock:
            bot._users["names"].clear()
        bot._end_conversation()

    def _feed(self, data):
        sock = mock.MagicMock(spec=socket.socket)
        chunks = [data, b""]
        sock.recv.side_effect = lambda size: chunks.pop(0) if chunks else b""
        thread = threading.Thread(target=bot.receiver, args=(sock,))
        thread.start()
        thread.join(timeout=2)

    def test_who_line_registers_member(self):
        line = (
            ":hive.2bd.net 352 {nick} #{chan} alice a.host hive.2bd.net "
            "alice (H) 0 :Alice".format(nick=bot.NICK, chan=bot.CHANNEL)
        )
        self._feed((line + "\r\n").encode())
        self.assertIn("alice", bot._channel_users())

    def test_name_reply_registers_members(self):
        line = ":hive.2bd.net 353 {nick} #{chan} :@alice +bob carol".format(
            nick=bot.NICK, chan=bot.CHANNEL
        )
        self._feed((line + "\r\n").encode())
        self.assertEqual(bot._channel_users(), ["alice", "bob", "carol"])


class TestMainJoinGrace(unittest.TestCase):
    """main() must record the join time so the join grace gate works.

    An early version wrote the join time into a local in main() with no way to
    reach the module global, so the global stayed 0.0 and
    `_within_join_grace()` was always False -- the bot opened its mouth before
    the userlist arrived and invented usernames. This runs the real main() with
    a mocked socket and asserts the global actually got the join time.
    """

    def setUp(self):
        with bot._prompt_lock:
            bot._joined["at"] = 0.0

    def tearDown(self):
        with bot._prompt_lock:
            bot._joined["at"] = 0.0

    def test_records_join_time_in_the_global(self):
        fake_socket = mock.MagicMock()
        fake_socket.return_value.recv.return_value = b""  # receiver exits cleanly
        with mock.patch.object(bot, "socket", new=fake_socket), \
             mock.patch.object(bot, "receiver"), \
             mock.patch.object(bot._registered, "wait", return_value=True), \
             mock.patch.object(bot, "_call_llm"), \
             mock.patch.object(bot.time, "sleep", side_effect=KeyboardInterrupt()):
            bot._registered.clear()
            t = threading.Thread(target=bot.main, daemon=True)
            t.start()
            t.join(timeout=5)
        with bot._prompt_lock:
            self.assertGreater(bot._joined["at"], 0.0)

    def test_within_grace_right_after_join(self):
        """Consequence of the above: the grace gate is True on a fresh join."""
        with bot._prompt_lock:
            bot._joined["at"] = time.monotonic()
        self.assertTrue(bot._within_join_grace())


class TestBotConstants(unittest.TestCase):
    """Test that constants are set correctly."""

    def test_server(self):
        self.assertEqual(bot.SERVER, "hive.2bd.net")

    def test_channel(self):
        self.assertEqual(bot.CHANNEL, "#hive")

    def test_nick(self):
        self.assertEqual(bot.NICK, "sloppy")


class TestLLMCallDebugRecord(unittest.TestCase):
    """The most recent LLM call is recorded so the TUI can inspect it (press 'd')."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()

    def test_call_records_system_prompt_messages_user_and_output(self):
        user_prompt = "what is 2+2?"
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        mock_response.choices[0].finish_reason = "stop"

        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            result = llmbot_core._call_llm(user_prompt)

        self.assertEqual(result, "The answer is 42.")
        messages = mock_create.call_args.kwargs["messages"]
        record = llmbot_core.get_last_llm_call()
        # system prompt, shown on its own
        self.assertIn("[System prompt]", record)
        self.assertIn(messages[0]["content"], record)
        # messages verbatim, in the same format they were sent to the model
        self.assertIn("[Messages]", record)
        self.assertIn(repr(messages), record)
        # the user prompt, shown on its own
        self.assertIn("[User message]", record)
        self.assertIn(user_prompt, record)
        # the returned output
        self.assertIn("[Output]", record)
        self.assertIn("The answer is 42.", record)

    def test_record_replaced_on_each_call(self):
        first = mock.MagicMock()
        first.choices = [mock.MagicMock()]
        first.choices[0].message.content = "FIRST ANSWER"
        first.choices[0].finish_reason = "stop"
        second = mock.MagicMock()
        second.choices = [mock.MagicMock()]
        second.choices[0].message.content = "SECOND ANSWER"
        second.choices[0].finish_reason = "stop"

        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=first
        ):
            llmbot_core._call_llm("first prompt")
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=second
        ):
            llmbot_core._call_llm("second prompt")

        record = llmbot_core.get_last_llm_call()
        self.assertIn("SECOND ANSWER", record)
        self.assertNotIn("FIRST ANSWER", record)

    def test_getter_is_safe_before_any_call(self):
        # No call has happened yet: the getter returns something displayable,
        # not an error.
        self.assertIsInstance(llmbot_core.get_last_llm_call(), str)


class TestSpeakRouting(unittest.TestCase):
    """A line the bot actually speaks is routed through the speak sink (blue),
    not the action sink (yellow)."""

    def setUp(self):
        with bot._prompt_lock:
            bot._pending["prompt"] = ""
        self._old_speak = llmbot_core.speak_sink
        self._old_action = llmbot_core.action_sink
        self._old_irc = llmbot_core.irc_sink
        self._speak_lines = []
        self._action_lines = []
        self._irc_lines = []
        llmbot_core.speak_sink = lambda m: self._speak_lines.append(m)
        llmbot_core.action_sink = lambda m: self._action_lines.append(m)
        llmbot_core.irc_sink = lambda m: self._irc_lines.append(m)

    def tearDown(self):
        llmbot_core.speak_sink = self._old_speak
        llmbot_core.action_sink = self._old_action
        llmbot_core.irc_sink = self._old_irc

    def test_reply_goes_to_speak_sink_not_action(self):
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=mock_response
        ):
            with llmbot_core._prompt_lock:
                llmbot_core._pending["prompt"] = "what is 2+2?"
            llmbot_core._process_pending(sock)

        self.assertTrue(
            any("The answer is 42." in m for m in self._speak_lines),
            "the bot's reply must go through the speak sink",
        )
        self.assertFalse(
            any("The answer is 42." in m for m in self._action_lines),
            "the bot's reply must not be styled as a generic action",
        )

    def test_status_lines_still_go_to_action_sink(self):
        # Being addressed / captured is still an action (yellow), not a speak.
        sock = mock.MagicMock(spec=socket.socket)
        self.assertTrue(
            llmbot_core._handle_ai_prompt(sock, "alice", "sloppy, hi")
        )
        self.assertTrue(
            any("Captured prompt" in m for m in self._action_lines)
        )
        self.assertFalse(
            any("Captured prompt" in m for m in self._speak_lines)
        )


class TestTUIStatusNote(unittest.TestCase):
    """The status pane advertises the LLM-call inspection view."""

    def test_status_shows_debug_option(self):
        import llmbot_tui

        snap = llmbot_core.status_snapshot()
        rendered = llmbot_tui._format_status(snap)
        # Indicators block stays free of the hint row...
        self.assertIn("Mood / Mode", rendered)
        self.assertNotIn("I = Inspect", rendered)
        # ...the hint row (pinned to the bottom of the status pane) advertises it.
        self.assertIn(
            "I = Inspect last LLM call", llmbot_tui._STATUS_HINTS
        )
        # The vision toggle is advertised alongside the other keys.
        self.assertIn("V = Toggle vision", llmbot_tui._STATUS_HINTS)


class TestVisionToggleTUI(unittest.IsolatedAsyncioTestCase):
    """The 'v' key cycles vision mode and the status line reflects it."""

    async def test_v_key_cycles_auto_on_off(self):
        import asyncio
        import llmbot_tui

        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None
            llmbot_core._vision["enabled"] = False

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                app.simulate_key("v")
                await asyncio.sleep(0.1)
                with llmbot_core._prompt_lock:
                    self.assertTrue(llmbot_core._vision["override"])
                app.simulate_key("v")
                await asyncio.sleep(0.1)
                with llmbot_core._prompt_lock:
                    self.assertFalse(llmbot_core._vision["override"])
                app.simulate_key("v")
                await asyncio.sleep(0.1)
                with llmbot_core._prompt_lock:
                    self.assertIsNone(llmbot_core._vision["override"])
        finally:
            llmbot_core.main = original_main
            with llmbot_core._prompt_lock:
                llmbot_core._vision["override"] = None

    async def test_status_shows_vision_line(self):
        import llmbot_tui

        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None
            llmbot_core._vision["enabled"] = True
        snap = llmbot_core.status_snapshot()
        rendered = llmbot_tui._format_status(snap)
        self.assertIn("Vision      :", rendered)
        self.assertIn("auto (enabled)", rendered)
        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None


class TestTUIStyleFixes(unittest.IsolatedAsyncioTestCase):
    """Three rendering fixes: speaks are blue+bold, the status hint row is
    pinned to the bottom of the status pane, and the debug modal wraps."""

    async def test_speak_is_blue_and_bold(self):
        import asyncio
        import llmbot_tui
        from textual.widgets import RichLog

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                log = app.query_one("#log", RichLog)
                app._on_speak("[AI] hi")
                await asyncio.sleep(0.1)
                style = list(log.lines[-1])[0].style
                self.assertTrue(style.bold)
                self.assertIn("bright_blue", str(style.color))
        finally:
            llmbot_core.main = original_main

    async def test_status_hints_pinned_to_bottom(self):
        import asyncio
        import llmbot_tui
        from textual.widgets import Static

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                hints = app.query_one("#status-hints", Static)
                # Docked to the bottom of the 40-row pane.
                self.assertEqual(hints.region.bottom, 40)
                info = app.query_one("#status-info", Static)
                # Indicators stay at the top, not overlapping the hint row.
                self.assertEqual(info.region.offset.y, 0)
                self.assertLess(info.region.bottom, hints.region.offset.y)
        finally:
            llmbot_core.main = original_main

    async def test_debug_modal_wraps_long_lines(self):
        import asyncio
        import llmbot_tui
        from textual.widgets import RichLog

        with llmbot_core._prompt_lock:
            llmbot_core._last_llm_call["text"] = "STYLE-TEST"
        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                app.simulate_key("i")
                await asyncio.sleep(0.1)
                dlog = ctx.app.screen.query_one("#llm_debug", RichLog)
                self.assertTrue(dlog.wrap)
                dlog.write("x" * 200)
                await asyncio.sleep(0.1)
                # A 200-char token must wrap into more than one line.
                self.assertGreater(len(dlog.lines), 1)
        finally:
            llmbot_core.main = original_main


class TestLLMDebugModal(unittest.IsolatedAsyncioTestCase):
    """'i'/'I' opens a scrollable modal of the last LLM call; it closes via
    Escape or the close button."""

    async def _with_modal(self, open_key, close):
        import asyncio
        import llmbot_tui

        with llmbot_core._prompt_lock:
            llmbot_core._last_llm_call["text"] = "MODAL-TEST-CALL"

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test() as ctx:
                ctx.app.simulate_key(open_key)
                await asyncio.sleep(0.1)
                screen = ctx.app.screen
                self.assertIsInstance(screen, llmbot_tui.LLMDebugView)
                self.assertTrue(screen.is_modal)
                self.assertEqual(screen.border_title, "Last LLM call")
                close(ctx.app)
                await asyncio.sleep(0.05)
                self.assertNotIsInstance(
                    ctx.app.screen, llmbot_tui.LLMDebugView
                )
        finally:
            llmbot_core.main = original_main

    async def test_i_opens_and_escapes_closes(self):
        await self._with_modal("i", lambda app: app.simulate_key("escape"))

    async def test_I_opens_and_escapes_closes(self):
        await self._with_modal("I", lambda app: app.simulate_key("escape"))

    async def test_button_closes(self):
        import llmbot_tui

        def close(app):
            app.screen.query_one("#close-btn", llmbot_tui.Button).press()

        await self._with_modal("i", close)


class TestExtractImageUrls(unittest.TestCase):
    """Image link detection in otherwise ordinary chat text."""

    def test_detects_extension_urls(self):
        urls = llmbot_core._extract_image_urls("look http://example.com/a/b/cat.jpg here")
        self.assertEqual(urls, ["http://example.com/a/b/cat.jpg"])

    def test_detects_multiple_and_png_webp(self):
        urls = llmbot_core._extract_image_urls("a https://x.io/p.png and y https://z.net/w.webp")
        self.assertEqual(urls, ["https://x.io/p.png", "https://z.net/w.webp"])

    def test_strips_trailing_punctuation(self):
        urls = llmbot_core._extract_image_urls("is that http://x.io/a.png?")
        self.assertEqual(urls, ["http://x.io/a.png"])

    def test_strips_trailing_brackets(self):
        urls = llmbot_core._extract_image_urls("(http://x.io/a.png)")
        self.assertEqual(urls, ["http://x.io/a.png"])

    def test_detects_extensionless_imgur(self):
        urls = llmbot_core._extract_image_urls("pic http://i.imgur.com/Ab12Cd")
        self.assertEqual(urls, ["http://i.imgur.com/Ab12Cd"])

    def test_ignores_non_image_links(self):
        self.assertEqual(llmbot_core._extract_image_urls("see http://example.com/article"), [])

    def test_first_image_url_none(self):
        self.assertIsNone(llmbot_core._first_image_url("no links here"))


class TestRecentImages(unittest.TestCase):
    """Per-nick / global most-recent image tracking."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_images["by_nick"].clear()
            llmbot_core._recent_images["global"] = None

    def test_records_and_reads_global(self):
        llmbot_core._record_image_url("tim", "http://img/t.jpg")
        self.assertEqual(llmbot_core._last_image_url(None), "http://img/t.jpg")

    def test_records_per_nick(self):
        llmbot_core._record_image_url("tim", "http://img/t.jpg")
        llmbot_core._record_image_url("jane", "http://img/j.jpg")
        self.assertEqual(llmbot_core._last_image_url("tim"), "http://img/t.jpg")
        self.assertEqual(llmbot_core._last_image_url("jane"), "http://img/j.jpg")

    def test_per_nick_case_insensitive(self):
        llmbot_core._record_image_url("Tim", "http://img/t.jpg")
        self.assertEqual(llmbot_core._last_image_url("tim"), "http://img/t.jpg")

    def test_records_on_note(self):
        llmbot_core._note_recent("check http://img/new.png", "tim")
        self.assertEqual(llmbot_core._last_image_url("tim"), "http://img/new.png")


class TestMatchVisionTrigger(unittest.TestCase):
    """On-demand image-request matching, command and referential forms."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_images["by_nick"].clear()
            llmbot_core._recent_images["global"] = None
            llmbot_core._users["names"].clear()

    def test_command_image(self):
        result = llmbot_core._match_vision_trigger("!image http://x.io/a.jpg")
        self.assertEqual(result[0], "http://x.io/a.jpg")
        self.assertEqual(result[2], llmbot_core.MODE_VISION)

    def test_command_img_with_colon(self):
        result = llmbot_core._match_vision_trigger("!img: http://x.io/a.png")
        self.assertEqual(result[0], "http://x.io/a.png")

    def test_command_image_prefix(self):
        result = llmbot_core._match_vision_trigger("image: http://x.io/a.png")
        self.assertEqual(result[0], "http://x.io/a.png")

    def test_command_prompt_is_remainder(self):
        result = llmbot_core._match_vision_trigger("!image http://x.io/a.jpg what is this")
        self.assertEqual(result[1], "what is this")

    def test_command_no_words_defaults_prompt(self):
        result = llmbot_core._match_vision_trigger("!image http://x.io/a.jpg")
        self.assertEqual(result[1], "what's in this image?")

    def test_command_no_url_is_not_vision(self):
        self.assertIsNone(llmbot_core._match_vision_trigger("!image what is this"))

    def test_referential_named_person(self):
        llmbot_core._register_user("Tim")
        llmbot_core._record_image_url("Tim", "http://img/t.jpg")
        result = llmbot_core._match_vision_trigger(f"{llmbot_core.NICK}, what's in the image Tim just posted?")
        self.assertEqual(result[0], "http://img/t.jpg")
        self.assertEqual(result[2], llmbot_core.MODE_VISION)

    def test_referential_global_when_no_nick(self):
        llmbot_core._record_image_url("tim", "http://img/global.jpg")
        result = llmbot_core._match_vision_trigger(f"{llmbot_core.NICK}, what's in the image just posted?")
        self.assertEqual(result[0], "http://img/global.jpg")

    def test_referential_no_url_falls_through(self):
        self.assertIsNone(llmbot_core._match_vision_trigger(f"{llmbot_core.NICK}, what's in the image?"))

    def test_plain_chat_is_not_vision(self):
        self.assertIsNone(llmbot_core._match_vision_trigger("hello everyone how's it going"))


class TestVisionActive(unittest.TestCase):
    """Auto-probe vs manual override resolution."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = False
            llmbot_core._vision["override"] = None

    def test_follows_probe_when_auto(self):
        llmbot_core._set_vision_override(None)
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = True
        self.assertTrue(llmbot_core._vision_active())
        self.assertEqual(llmbot_core._vision_source(), "auto")

    def test_override_on_beats_probe(self):
        llmbot_core._set_vision_override(True)
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = False
        self.assertTrue(llmbot_core._vision_active())
        self.assertEqual(llmbot_core._vision_source(), "on")

    def test_override_off_beats_probe(self):
        llmbot_core._set_vision_override(False)
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = True
        self.assertFalse(llmbot_core._vision_active())
        self.assertEqual(llmbot_core._vision_source(), "off")

    def test_cycle_auto_on_off(self):
        self.assertEqual(llmbot_core._cycle_vision_override(), "on")
        self.assertTrue(llmbot_core._vision_active())
        self.assertEqual(llmbot_core._cycle_vision_override(), "off")
        self.assertFalse(llmbot_core._vision_active())
        self.assertEqual(llmbot_core._cycle_vision_override(), "auto")
        self.assertEqual(llmbot_core._vision_source(), "auto")


class TestProbeVision(unittest.TestCase):
    """/props probing for modalities.vision."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = False
            llmbot_core._vision["override"] = None

    def _fake_resp(self, payload):
        import json as _json
        resp = mock.MagicMock()
        resp.read.return_value = _json.dumps(payload).encode("utf-8")
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = resp
        ctx.__exit__.return_value = False
        return ctx

    def test_detects_vision_true(self):
        ctx = self._fake_resp({"modalities": {"vision": True}})
        with mock.patch("urllib.request.urlopen", return_value=ctx):
            self.assertTrue(llmbot_core._probe_vision())
        with llmbot_core._prompt_lock:
            self.assertTrue(llmbot_core._vision["enabled"])

    def test_detects_vision_false(self):
        ctx = self._fake_resp({"modalities": {"vision": False}})
        with mock.patch("urllib.request.urlopen", return_value=ctx):
            self.assertFalse(llmbot_core._probe_vision())

    def test_probe_failure_is_not_enabled(self):
        with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("down")):
            self.assertFalse(llmbot_core._probe_vision())


class TestCallLLMVision(unittest.TestCase):
    """Image rides on the user message as an image_url content part."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()

    def test_builds_image_content_array(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "A tabby cat on a mat"
        mock_response.choices[0].finish_reason = "stop"

        with mock.patch.object(llmbot_core._llm_client.chat.completions, "create",
                               return_value=mock_response) as mock_create:
            result = llmbot_core._call_llm_vision("http://x.io/a.jpg", "what's this?")

        self.assertEqual(result, "A tabby cat on a mat")
        messages = mock_create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        user = messages[-1]
        self.assertEqual(user["role"], "user")
        content = user["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "what's this?")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "http://x.io/a.jpg")

    def test_injects_recent_history(self):
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "seen"
        mock_response.choices[0].finish_reason = "stop"
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.append("alice: hi")
            llmbot_core._recent_senders.append("alice")
        with mock.patch.object(llmbot_core._llm_client.chat.completions, "create",
                               return_value=mock_response) as mock_create:
            llmbot_core._call_llm_vision("http://x.io/a.jpg", "what?")
        messages = mock_create.call_args.kwargs["messages"]
        # The recent line sits between the system prompt and the image user
        # message, so the model sees the reply as spoken into an ongoing room.
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("alice: hi", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "user")
        self.assertIsInstance(messages[2]["content"], list)


class TestHandleAIImage(unittest.TestCase):
    """End-to-end capture and answering of on-demand image requests."""

    def setUp(self):
        llmbot_core._end_conversation()
        with llmbot_core._prompt_lock:
            llmbot_core._pending["prompt"] = ""
            llmbot_core._pending_vision["url"] = ""
            llmbot_core._vision["enabled"] = False
            llmbot_core._vision["override"] = None

    def test_command_queued_when_active(self):
        llmbot_core._set_vision_override(True)
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_ai_prompt(sock, "alice", "!image http://x.io/a.jpg what is this")
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._pending_vision["url"], "http://x.io/a.jpg")
            self.assertEqual(llmbot_core._pending_vision["prompt"], "what is this")
        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None

    def test_refused_when_not_active(self):
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_ai_prompt(sock, "alice", "!image http://x.io/a.jpg")
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._pending_vision["url"], "")
        self.assertEqual(sock.send.call_count, 1)
        sent = sock.send.call_args.args[0].decode("utf-8")
        self.assertIn("can't see images", sent)

    def test_process_pending_vision_answers(self):
        llmbot_core._set_vision_override(True)
        llmbot_core._queue_vision("http://x.io/a.jpg", "alice", "what is this")
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "A cat"
        mock_response.choices[0].finish_reason = "stop"
        with mock.patch.object(llmbot_core._llm_client.chat.completions, "create",
                               return_value=mock_response):
            llmbot_core._process_pending_vision(sock)
        self.assertEqual(sock.send.call_count, 1)
        sent = sock.send.call_args.args[0].decode("utf-8")
        self.assertIn("A cat", sent)
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._pending_vision["url"], "")
        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None


class TestStatusSnapshotVision(unittest.TestCase):
    """Vision state is surfaced in the status snapshot."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._vision["enabled"] = True
            llmbot_core._vision["override"] = None

    def test_snapshot_defaults_auto(self):
        snap = llmbot_core.status_snapshot()
        self.assertTrue(snap["vision"])
        self.assertEqual(snap["vision_source"], "auto")

    def test_snapshot_manual_on(self):
        llmbot_core._set_vision_override(True)
        snap = llmbot_core.status_snapshot()
        self.assertEqual(snap["vision_source"], "on")
        with llmbot_core._prompt_lock:
            llmbot_core._vision["override"] = None


if __name__ == "__main__":
    unittest.main()


class TestSplitEvent(unittest.TestCase):
    """Split an IRC event line into (nick, COMMAND)."""

    def test_join(self):
        self.assertEqual(llmbot_core._split_event(":alice!u@h JOIN #hive"), ("alice", "JOIN"))

    def test_join_without_channel(self):
        self.assertEqual(llmbot_core._split_event(":alice!u@h JOIN"), ("alice", "JOIN"))

    def test_quit(self):
        self.assertEqual(llmbot_core._split_event(":alice!u@h QUIT :bye"), ("alice", "QUIT"))

    def test_part(self):
        self.assertEqual(llmbot_core._split_event(":bob!u@h PART #hive :cya"), ("bob", "PART"))

    def test_privmsg_is_not_an_event(self):
        self.assertEqual(llmbot_core._split_event(":alice!u@h PRIVMSG #hive :hi"), ("alice", "PRIVMSG"))

    def test_non_event_line(self):
        self.assertEqual(llmbot_core._split_event("no colon here"), ("", ""))


class TestGreetingText(unittest.TestCase):
    """The greeting pool, roast chance, and nick insertion."""

    def test_greeting_includes_nick(self):
        text = llmbot_core._join_greeting_text("SpecialNick")
        self.assertIsNotNone(text)
        self.assertIn("SpecialNick", text)

    def test_roast_added_when_chance_high(self):
        # A roast is added when random() < GREET_ROAST_CHANCE, so a low draw
        # forces the roast.
        with mock.patch.object(random, "random", return_value=0.0), \
             mock.patch.object(random, "choice", return_value="WELCOME"):
            text = llmbot_core._greeting_text("join", "x")
        self.assertEqual(text, "WELCOME WELCOME")

    def test_no_roast_when_chance_low(self):
        # A high draw (>= the chance) skips the roast.
        with mock.patch.object(random, "random", return_value=1.0), \
             mock.patch.object(random, "choice", return_value="WELCOME"):
            text = llmbot_core._greeting_text("return", "x")
        self.assertEqual(text, "WELCOME")


class TestJoinGreet(unittest.TestCase):
    """A JOIN greets the newcomer, unless they only just left."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at.clear()
            llmbot_core._chatlines["count"] = 0
            llmbot_core._last_seen.clear()

    def test_newcomer_is_greeted(self):
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_join(sock, "newbie")
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertTrue(any(b"PRIVMSG" in s and b"newbie" in s for s in sends))

    def test_bot_join_is_not_greeted(self):
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_join(sock, llmbot_core.NICK)
        self.assertEqual(sock.send.call_count, 0)

    def test_skip_when_left_recently(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at["popper"] = 3
            llmbot_core._chatlines["count"] = 5  # 5 - 3 = 2 chatlines since leave
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_join(sock, "popper")
        self.assertEqual(sock.send.call_count, 0)

    def test_greet_after_longer_gap(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at["popper"] = 0
            llmbot_core._chatlines["count"] = 5  # 5 chatlines since leave, not < 5
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_join(sock, "popper")
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertTrue(any(b"PRIVMSG" in s for s in sends))

    def test_join_resets_idle_timer(self):
        sock = mock.MagicMock(spec=socket.socket)
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["x"] = 0.0
        llmbot_core._handle_join(sock, "x")
        with llmbot_core._prompt_lock:
            self.assertGreater(llmbot_core._last_seen["x"], 0.0)


class TestQuitTracking(unittest.TestCase):
    """A QUIT/PART records the exit chatline so a quick rejoin is skipped."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at.clear()
            llmbot_core._chatlines["count"] = 10

    def test_quit_records_chatline(self):
        llmbot_core._handle_quit("alice")
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._left_at["alice"], 10)

    def test_skip_greeting_within_five_chatlines(self):
        llmbot_core._handle_quit("alice")
        with llmbot_core._prompt_lock:
            llmbot_core._chatlines["count"] = 12  # only 2 chatlines since leave
        self.assertIsNone(llmbot_core._join_greeting_text("alice"))

    def test_greet_after_five_chatlines(self):
        llmbot_core._handle_quit("alice")
        with llmbot_core._prompt_lock:
            llmbot_core._chatlines["count"] = 20  # 10 chatlines since leave
        self.assertIsNotNone(llmbot_core._join_greeting_text("alice"))

    def test_quit_ignores_empty_nick(self):
        llmbot_core._handle_quit("")
        with llmbot_core._prompt_lock:
            self.assertNotIn("", llmbot_core._left_at)


class TestIdleGreet(unittest.TestCase):
    """A message after IDLE_GREET_AFTER of silence is welcomed back once."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen.clear()
            llmbot_core._chatlines["count"] = 0

    def test_greet_after_long_idle(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["alice"] = time.monotonic() - (llmbot_core.IDLE_GREET_AFTER + 10)
        greeting = llmbot_core._note_recent("hi there", "alice")
        self.assertIsNotNone(greeting)
        self.assertIn("alice", greeting)

    def test_no_greet_for_recent_message(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["alice"] = time.monotonic() - 5
        self.assertIsNone(llmbot_core._note_recent("hi there", "alice"))

    def test_no_greet_for_first_message(self):
        self.assertIsNone(llmbot_core._note_recent("hi there", "alice"))

    def test_timer_resets_after_greeting(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["alice"] = time.monotonic() - (llmbot_core.IDLE_GREET_AFTER + 10)
        self.assertIsNotNone(llmbot_core._note_recent("first", "alice"))
        self.assertIsNone(llmbot_core._note_recent("second", "alice"))

    def test_bot_own_message_not_tracked(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen[llmbot_core.NICK] = 0.0
        self.assertIsNone(llmbot_core._note_recent("echo", llmbot_core.NICK))
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._last_seen[llmbot_core.NICK], 0.0)


class TestReceiverGreetIntegration(unittest.TestCase):
    """The receiver wires JOIN/QUIT/idle into greetings."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at.clear()
            llmbot_core._chatlines["count"] = 0
            llmbot_core._last_seen.clear()
            llmbot_core._pending["prompt"] = ""
        llmbot_core._end_conversation()

    def _feed(self, line):
        sock = mock.MagicMock(spec=socket.socket)
        payload = (line + "\r\n").encode()
        chunks = [payload, b""]
        sock.recv.side_effect = lambda size: chunks.pop(0) if chunks else b""
        t = threading.Thread(target=llmbot_core.receiver, args=(sock,))
        t.start()
        t.join(timeout=2)
        return [c.args[0] for c in sock.send.call_args_list]

    def test_join_line_sends_greeting(self):
        sends = self._feed(":newbie!u@h JOIN #hive")
        self.assertTrue(any(b"PRIVMSG" in s and b"newbie" in s for s in sends))

    def test_quit_line_records_exit(self):
        self._feed(":bob!u@h QUIT :bye")
        with llmbot_core._prompt_lock:
            self.assertIn("bob", llmbot_core._left_at)

    def test_idle_message_sends_greeting(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["alice"] = time.monotonic() - (llmbot_core.IDLE_GREET_AFTER + 10)
        sends = self._feed(":alice!u@h PRIVMSG #hive :back already?")
        self.assertTrue(any(b"PRIVMSG" in s for s in sends))


class TestTrivialMessageFilter(unittest.TestCase):
    """Lines that are a single word or shorter than MIN_CHAT_CHARS are not
    stored in the LLM's recent-history buffer (but still count/track)."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()
            llmbot_core._last_seen.clear()
            llmbot_core._chatlines["count"] = 0

    def test_short_line_is_trivial(self):
        self.assertTrue(llmbot_core._is_trivial_message("hi"))

    def test_single_long_word_is_trivial(self):
        # A single word is trivial regardless of length.
        self.assertTrue(llmbot_core._is_trivial_message("supercalifragilistic"))

    def test_multi_word_line_is_stored(self):
        self.assertFalse(llmbot_core._is_trivial_message("hello world"))

    def test_trivial_message_not_stored(self):
        llmbot_core._note_recent("lol", "alice")
        with llmbot_core._prompt_lock:
            self.assertEqual(len(llmbot_core._recent_lines), 0)
            self.assertEqual(len(llmbot_core._recent_senders), 0)

    def test_real_message_is_stored(self):
        llmbot_core._note_recent("what do you think about this", "alice")
        with llmbot_core._prompt_lock:
            self.assertEqual(len(llmbot_core._recent_lines), 1)
            self.assertEqual(llmbot_core._recent_lines[0], "what do you think about this")
            self.assertEqual(llmbot_core._recent_senders[0], "alice")

    def test_trivial_message_still_counts_as_chatline(self):
        llmbot_core._note_recent("lol", "alice")
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._chatlines["count"], 1)

    def test_trivial_message_still_updates_last_seen(self):
        with llmbot_core._prompt_lock:
            llmbot_core._last_seen["alice"] = 0.0
        llmbot_core._note_recent("lol", "alice")
        with llmbot_core._prompt_lock:
            self.assertGreater(llmbot_core._last_seen["alice"], 0.0)


class TestPause(unittest.TestCase):
    """Pressing 'P' pauses the bot: no LLM calls and no greetings until 'P'
    is pressed again to unpause."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._left_at.clear()
            llmbot_core._chatlines["count"] = 0
            llmbot_core._last_seen.clear()
            llmbot_core._paused["on"] = False
            llmbot_core._pending["prompt"] = ""

    def test_toggle_flips_state(self):
        llmbot_core._toggle_pause()
        with llmbot_core._prompt_lock:
            self.assertTrue(llmbot_core._paused["on"])
        llmbot_core._toggle_pause()
        with llmbot_core._prompt_lock:
            self.assertFalse(llmbot_core._paused["on"])

    def test_paused_process_pending_makes_no_call(self):
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "answer"
        with llmbot_core._prompt_lock:
            llmbot_core._paused["on"] = True
            llmbot_core._pending["prompt"] = "what is 2+2?"
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=mock_response
        ) as create:
            llmbot_core._process_pending(sock)
        create.assert_not_called()
        self.assertEqual(sock.send.call_count, 0)

    def test_unpaused_process_pending_calls_llm(self):
        sock = mock.MagicMock(spec=socket.socket)
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "The answer is 42."
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions, "create", return_value=mock_response
        ):
            with llmbot_core._prompt_lock:
                llmbot_core._pending["prompt"] = "what is 2+2?"
            llmbot_core._process_pending(sock)
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertTrue(any(b"PRIVMSG #hive :The answer is 42." in s for s in sends))

    def test_paused_no_join_greeting(self):
        sock = mock.MagicMock(spec=socket.socket)
        with llmbot_core._prompt_lock:
            llmbot_core._paused["on"] = True
        llmbot_core._handle_join(sock, "newbie")
        self.assertEqual(sock.send.call_count, 0)

    def test_unpaused_join_greeting(self):
        sock = mock.MagicMock(spec=socket.socket)
        llmbot_core._handle_join(sock, "newbie")
        sends = [c.args[0] for c in sock.send.call_args_list]
        self.assertTrue(any(b"PRIVMSG" in s and b"newbie" in s for s in sends))

    def test_paused_no_idle_greeting(self):
        with llmbot_core._prompt_lock:
            llmbot_core._paused["on"] = True
            llmbot_core._last_seen["alice"] = time.monotonic() - (llmbot_core.IDLE_GREET_AFTER + 10)
        greeting = llmbot_core._note_recent("back already", "alice")
        self.assertIsNone(greeting)


class TestPauseTUI(unittest.IsolatedAsyncioTestCase):
    """The 'P' key toggles pause, and the hint row advertises it."""

    async def test_p_key_toggles_pause(self):
        import asyncio
        import llmbot_tui

        with llmbot_core._prompt_lock:
            llmbot_core._paused["on"] = False

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                app.simulate_key("p")
                await asyncio.sleep(0.1)
                with llmbot_core._prompt_lock:
                    self.assertTrue(llmbot_core._paused["on"])
                app.simulate_key("p")
                await asyncio.sleep(0.1)
                with llmbot_core._prompt_lock:
                    self.assertFalse(llmbot_core._paused["on"])
        finally:
            llmbot_core.main = original_main
            with llmbot_core._prompt_lock:
                llmbot_core._paused["on"] = False

    async def test_hint_row_advertises_pause(self):
        import llmbot_tui

        self.assertIn("P to pause", llmbot_tui._STATUS_HINTS)




class TestSummarizerIntegration(unittest.TestCase):
    """The rolling summarizer integrated into llmbot_core: the unsummarized-line
    buffer, the background worker, and the summary/highlights prompt context.
    summarize_tick is mocked so no server is touched; the mock returns a
    controlled (summary, highlights) tuple."""

    def setUp(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.clear()
            llmbot_core._recent_senders.clear()
            llmbot_core._pending_summary_lines.clear()
            llmbot_core._rolling["summary"] = ""
            llmbot_core._rolling["highlights"] = []
            llmbot_core._last_summary_at = {"t": 0.0}
            llmbot_core._paused["on"] = False

    def tearDown(self):
        # Never leak rolling state into later tests.
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.clear()
            llmbot_core._rolling["summary"] = ""
            llmbot_core._rolling["highlights"] = []
            llmbot_core._paused["on"] = False

    def test_new_line_goes_to_both_buffers(self):
        llmbot_core._note_recent("hello there everyone", "alice")
        with llmbot_core._prompt_lock:
            self.assertIn("hello there everyone", list(llmbot_core._recent_lines))
            self.assertIn(
                "hello there everyone", list(llmbot_core._pending_summary_lines)
            )

    def test_pending_skips_own_nick_and_trivial(self):
        # The bot's own echoes are skipped, mirroring the chatter buffer.
        llmbot_core._note_recent("hi there pal", llmbot_core.NICK)
        with llmbot_core._prompt_lock:
            self.assertEqual(list(llmbot_core._pending_summary_lines), [])
        # Lines shorter than MIN_CHAT_CHARS are not stored either.
        llmbot_core._note_recent("lol", "alice")
        with llmbot_core._prompt_lock:
            self.assertEqual(list(llmbot_core._pending_summary_lines), [])

    def test_worker_summarizes_on_age_trigger(self):
        # Age arm: more than SUMMARIZE_INTERVAL since the last summary, with
        # enough lines to clear the minimum.
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.extend(
                [f"n{i}: line{i}" for i in range(6)]
            )
            llmbot_core._rolling["summary"] = "old"
            llmbot_core._rolling["highlights"] = ["old quote"]
            llmbot_core._last_summary_at["t"] = time.monotonic() - (
                llmbot_core.SUMMARIZE_INTERVAL + 60
            )
        with mock.patch.object(
            llmbot_core.summarizer,
            "summarize_tick",
            return_value=("rolling summary", ["first quote", "second quote"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(list(llmbot_core._pending_summary_lines), [])
            self.assertEqual(llmbot_core._rolling["summary"], "rolling summary")
            self.assertEqual(
                llmbot_core._rolling["highlights"], ["first quote", "second quote"]
            )
            self.assertGreater(llmbot_core._last_summary_at["t"], 0.0)

    def test_worker_is_noop_when_no_pending(self):
        with llmbot_core._prompt_lock:
            llmbot_core._rolling["summary"] = "keep"
            llmbot_core._rolling["highlights"] = ["k"]
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick", return_value=("x", ["y"])
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "keep")
            self.assertEqual(llmbot_core._rolling["highlights"], ["k"])

    def test_worker_skipped_while_paused(self):
        # Pause overrides a valid trigger: age is old and there are enough lines,
        # yet nothing is summarized.
        lines = [f"n{i}: line{i}" for i in range(6)]
        with llmbot_core._prompt_lock:
            llmbot_core._paused["on"] = True
            llmbot_core._pending_summary_lines.extend(lines)
            llmbot_core._last_summary_at["t"] = time.monotonic() - (
                llmbot_core.SUMMARIZE_INTERVAL + 60
            )
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick", return_value=("x", ["y"])
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(list(llmbot_core._pending_summary_lines), lines)
            self.assertEqual(llmbot_core._rolling["summary"], "")

    def test_snapshot_is_independent_of_live_list(self):
        # The worker snapshots then clears the live list; a line arriving while
        # the LLM generates must not leak into the summarizer's input.
        seen = {}

        def fake(prev_summary, prev_highlights, lines):
            with llmbot_core._prompt_lock:
                seen["snapshot"] = list(lines)
                llmbot_core._pending_summary_lines.append("alice: during")
            return ("s", ["h"])

        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick", side_effect=fake
        ):
            with llmbot_core._prompt_lock:
                llmbot_core._pending_summary_lines.extend(
                    [f"n{i}: line{i}" for i in range(6)]
                )
                llmbot_core._last_summary_at["t"] = time.monotonic() - (
                    llmbot_core.SUMMARIZE_INTERVAL + 60
                )
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(
                seen["snapshot"], [f"n{i}: line{i}" for i in range(6)]
            )
            self.assertEqual(list(llmbot_core._pending_summary_lines), ["alice: during"])

    def test_worker_summarizes_on_volume_trigger(self):
        # Volume arm: recent summary, but more than SUMMARIZE_VOLUME_LINES lines.
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.extend(
                [f"n{i}: line{i}"
                 for i in range(llmbot_core.SUMMARIZE_VOLUME_LINES + 1)]
            )
            llmbot_core._last_summary_at["t"] = time.monotonic()
        with mock.patch.object(
            llmbot_core.summarizer,
            "summarize_tick",
            return_value=("rolled", ["v quote"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(list(llmbot_core._pending_summary_lines), [])
            self.assertEqual(llmbot_core._rolling["summary"], "rolled")

    def test_worker_gates_on_minimum_lines(self):
        # Old enough to trigger on age, but fewer than SUMMARIZE_MIN_LINES lines.
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.extend(
                [f"n{i}: line{i}"
                 for i in range(llmbot_core.SUMMARIZE_MIN_LINES - 1)]
            )
            llmbot_core._last_summary_at["t"] = time.monotonic() - (
                llmbot_core.SUMMARIZE_INTERVAL + 60
            )
        with mock.patch.object(
            llmbot_core.summarizer,
            "summarize_tick",
            return_value=("rolled", ["q"]),
        ) as p:
            llmbot_core._summarize_pending()
        p.assert_not_called()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "")

    def test_worker_no_trigger_midrange(self):
        # Recent summary and between MIN and VOLUME lines: neither arm fires.
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.extend(
                [f"n{i}: line{i}" for i in range(20)]
            )
            llmbot_core._last_summary_at["t"] = time.monotonic()
        with mock.patch.object(
            llmbot_core.summarizer,
            "summarize_tick",
            return_value=("rolled", ["q"]),
        ) as p:
            llmbot_core._summarize_pending()
        p.assert_not_called()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "")
            self.assertEqual(
                list(llmbot_core._pending_summary_lines),
                [f"n{i}: line{i}" for i in range(20)],
            )

    def test_call_llm_injects_summary_then_recent(self):
        with llmbot_core._prompt_lock:
            llmbot_core._rolling["summary"] = "the channel discussed the launch"
            llmbot_core._rolling["highlights"] = ["one memorable quote"]
            llmbot_core._recent_lines.extend(["a", "b"])
            llmbot_core._recent_senders.extend(["alice", "bob"])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "ok"
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions,
            "create",
            return_value=mock_response,
        ) as create:
            llmbot_core._call_llm("hey")
        messages = create.call_args.kwargs["messages"]
        # persona system, then ONE system context block, then the user input.
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "system")
        context = messages[1]["content"]
        self.assertIn("--- CONVERSATION MEMORY ---", context)
        self.assertIn("the channel discussed the launch", context)
        self.assertIn("--- IMPORTANT HIGHLIGHTS ---", context)
        self.assertIn("one memorable quote", context)
        self.assertIn("--- RECENT IRC CHAT ---", context)
        self.assertIn("alice: a", context)
        self.assertIn("bob: b", context)
        # The recent lines live inside the context block, not as separate user
        # messages; the current input is the only user message.
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "hey")

    def test_call_llm_without_summary_injects_only_recent(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.extend(["a"])
            llmbot_core._recent_senders.extend(["alice"])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "ok"
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions,
            "create",
            return_value=mock_response,
        ) as create:
            llmbot_core._call_llm("hey")
        messages = create.call_args.kwargs["messages"]
        # No summary/highlights yet: only the recent chat section in the system
        # context block, then the user input.
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("--- RECENT IRC CHAT ---", messages[1]["content"])
        self.assertIn("alice: a", messages[1]["content"])
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["content"], "hey")

    def test_call_llm_caps_recent_chat_at_20(self):
        with llmbot_core._prompt_lock:
            llmbot_core._recent_lines.extend(f"line{i}" for i in range(100))
            llmbot_core._recent_senders.extend([f"n{i}" for i in range(100)])
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "ok"
        with mock.patch.object(
            llmbot_core._llm_client.chat.completions,
            "create",
            return_value=mock_response,
        ) as create:
            llmbot_core._call_llm("hey")
        messages = create.call_args.kwargs["messages"]
        # persona system + one system context block + user.
        self.assertEqual(len(messages), 3)
        context = messages[1]["content"]
        # Only the last 20 of the 100 lines are injected.
        self.assertNotIn("n0: line0", context)
        self.assertIn("n80: line80", context)
        self.assertIn("n99: line99", context)

    def test_summarize_tick_returns_inputs_on_server_error(self):
        # The module-level contract: never raise, return inputs unchanged.
        import summarizer

        with mock.patch.object(
            summarizer.requests, "post", side_effect=RuntimeError("server down")
        ):
            out = summarizer.summarize_tick("old", ["hq"], ["x: y"])
        self.assertEqual(out, ("old", ["hq"]))


class TestRejectReason(unittest.TestCase):
    """A summary is usable only if it is a non-empty string within the cap."""

    def test_valid_summary_accepted(self):
        self.assertIsNone(llmbot_core._reject_reason("a real summary"))

    def test_whitespace_only_is_empty(self):
        self.assertEqual(
            llmbot_core._reject_reason("   "), "summary was empty"
        )

    def test_non_string_rejected(self):
        self.assertEqual(
            llmbot_core._reject_reason(None), "summary was not a string"
        )
        self.assertEqual(
            llmbot_core._reject_reason(["nope"]), "summary was not a string"
        )
        self.assertEqual(
            llmbot_core._reject_reason(42), "summary was not a string"
        )

    def test_at_cap_is_accepted(self):
        # Exactly the cap is fine; only over it is rejected.
        summary = "x" * llmbot_core.SUMMARIZE_MAX_CHARS
        self.assertIsNone(llmbot_core._reject_reason(summary))

    def test_oversized_summary_rejected(self):
        summary = "x" * (llmbot_core.SUMMARIZE_MAX_CHARS + 1)
        reason = llmbot_core._reject_reason(summary)
        self.assertIsNotNone(reason)
        self.assertIn("exceeds", reason)
        self.assertIn(str(llmbot_core.SUMMARIZE_MAX_CHARS), reason)


class TestSummaryValidation(unittest.TestCase):
    """An unusable summary from the model is rejected, not stored.

    The previous rolling summary is kept and a warning is emitted; a valid
    summary is stored as before with no warning.
    """

    def setUp(self):
        self._old_warning = llmbot_core.warning
        self._warnings = []
        llmbot_core.warning = lambda m: self._warnings.append(m)

    def tearDown(self):
        llmbot_core.warning = self._old_warning

    def _seed(self, summary, highlights, pending):
        with llmbot_core._prompt_lock:
            llmbot_core._pending_summary_lines.extend(pending)
            llmbot_core._rolling["summary"] = summary
            llmbot_core._rolling["highlights"] = highlights
            llmbot_core._last_summary_at["t"] = llmbot_core.time.monotonic() - (
                llmbot_core.SUMMARIZE_INTERVAL + 60
            )

    def test_empty_summary_keeps_previous(self):
        self._seed("OLD", ["h"], [f"n{i}: l{i}" for i in range(6)])
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick",
            return_value=("", ["new h"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "OLD")
        self.assertEqual(len(self._warnings), 1)
        self.assertTrue(
            self._warnings[0].startswith("INVALID SUMMARY RECEIVED:")
        )

    def test_oversized_summary_keeps_previous(self):
        big = "x" * (llmbot_core.SUMMARIZE_MAX_CHARS + 1)
        self._seed("OLD", ["h"], [f"n{i}: l{i}" for i in range(6)])
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick",
            return_value=(big, ["new h"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "OLD")
        self.assertEqual(len(self._warnings), 1)

    def test_non_string_summary_keeps_previous(self):
        self._seed("OLD", ["h"], [f"n{i}: l{i}" for i in range(6)])
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick",
            return_value=(42, ["new h"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "OLD")
        self.assertEqual(len(self._warnings), 1)

    def test_valid_summary_stored_and_no_warning(self):
        self._seed("OLD", ["h"], [f"n{i}: l{i}" for i in range(6)])
        with mock.patch.object(
            llmbot_core.summarizer, "summarize_tick",
            return_value=("NEW", ["new h"]),
        ):
            llmbot_core._summarize_pending()
        with llmbot_core._prompt_lock:
            self.assertEqual(llmbot_core._rolling["summary"], "NEW")
        self.assertEqual(self._warnings, [])


class TestWarningRendering(unittest.IsolatedAsyncioTestCase):
    """A rejected summary is written to the log pane in red, not yellow."""

    async def test_warning_is_red(self):
        import asyncio
        import llmbot_tui
        from textual.widgets import RichLog

        original_main = llmbot_core.main
        llmbot_core.main = lambda *a, **k: None
        try:
            app = llmbot_tui.LLMBotApp()
            async with app.run_test(size=(120, 40)) as ctx:
                log = app.query_one("#log", RichLog)
                app._on_warning("[AI] INVALID SUMMARY RECEIVED: summary was empty")
                await asyncio.sleep(0.1)
                style = list(log.lines[-1])[0].style
                self.assertTrue(style.bold)
                self.assertIn("red", str(style.color))
        finally:
            llmbot_core.main = original_main
