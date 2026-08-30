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
                         bot._system_prompt(bot.MODE_SERIOUS))

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
