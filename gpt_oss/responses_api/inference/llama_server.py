"""Inference backend using a local `llama.cpp` server.

This backend connects to a running `llama.cpp` HTTP server (the
`llama-server` binary) and streams generated tokens. The server is
expected to expose an OpenAI-compatible `/v1/chat/completions` endpoint
with streaming enabled.
"""

import json
import threading
import time
from typing import Callable, Optional

import os
import requests
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

# Special token emitted by other inference backends to indicate end of stream
EOS_TOKEN = 200002

# Tunables controlling polling / timeout behaviour
POLL_INTERVAL_S = 0.01
CALL_MAX_WAIT_S = 0.250
NO_TOKEN_TIMEOUT_S = 15.0
FIRST_BYTE_TIMEOUT_S = 30.0

# Shared mutable state used between calls
_token_buffer: list[int] = []
_buffer_lock = threading.Lock()
_stream_thread: Optional[threading.Thread] = None
_stream_done = threading.Event()
_stream_error: Optional[Exception] = None
_last_progress_ts: float = 0.0


def _now() -> float:
    return time.monotonic()


def _touch_progress() -> None:
    global _last_progress_ts
    _last_progress_ts = _now()


def _reset_stream_state() -> None:
    global _token_buffer, _stream_thread, _stream_error
    with _buffer_lock:
        _token_buffer = []
    _stream_done.clear()
    _stream_thread = None
    _stream_error = None
    _touch_progress()


def setup_model(_checkpoint: str) -> Callable[[list[int], float, bool], int]:
    """Return a token-by-token inference function using llama-server."""
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Allow overriding server base URL and model via environment; derive sensible defaults.
    base_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080").rstrip("/")
    # Derive a model name from checkpoint path if provided; fallback to a generic name.
    derived_model = (
        os.environ.get("LLAMA_SERVER_MODEL")
        or os.path.basename(os.path.expanduser(_checkpoint or "")).strip() or "gpt-oss"
    )

    def _harmony_to_openai_messages(prompt_text: str) -> list[dict]:
        """
        Convert Harmony-rendered conversation text (with <|start|>, <|message|>, <|end|>)
        into an OpenAI chat messages array.
        We keep only roles system/user/assistant and collapse any extra channel markers.
        Ensures the final turn is an assistant message with empty content to elicit a completion.
        """
        START = "<|start|>"
        END = "<|end|>"
        MESSAGE = "<|message|>"
        CHANNEL = "<|channel|>"

        s = prompt_text
        i = 0
        messages: list[dict] = []
        while True:
            start_idx = s.find(START, i)
            if start_idx < 0:
                break
            end_idx = s.find(END, start_idx + len(START))
            if end_idx < 0:
                break
            segment = s[start_idx + len(START) : end_idx]
            i = end_idx + len(END)

            seg = segment.strip()
            if not seg:
                continue

            # Split header (role and optional metadata) from content.
            if MESSAGE in seg:
                header, content = seg.split(MESSAGE, 1)
            else:
                # If there is no explicit <|message|>, try splitting on first newline.
                if "\n" in seg:
                    header, content = seg.split("\n", 1)
                else:
                    header, content = seg, ""

            header = header.strip()
            content = content.replace(CHANNEL, "").strip()

            # Normalize role
            role = header.strip()
            if role not in ("system", "user", "assistant"):
                if role == "developer":
                    role = "system"
                else:
                    # Unknown roles are coerced into system notes
                    role = "system"

            messages.append({"role": role, "content": content})

        # Ensure the last message primes an assistant turn
        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content"):
            messages.append({"role": "assistant", "content": ""})

        return messages

    def _start_stream(token_ids: list[int], temperature: float) -> threading.Thread:
        prompt_text = encoding.decode(token_ids)

        def run() -> None:
            nonlocal prompt_text, temperature
            global _stream_error
            accum_text = ""
            last_len = 0
            try:
                # Use OpenAI-compatible chat completions streaming endpoint
                url = f"{base_url}/v1/chat/completions"
                payload = {
                    "model": derived_model,
                    "messages": _harmony_to_openai_messages(prompt_text),
                    "temperature": float(temperature),
                    "stream": True,
                }
                with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = line.strip()
                        if data.startswith("data:"):
                            data = data[len("data:") :].strip()
                        if data == "[DONE]":
                            with _buffer_lock:
                                _token_buffer.append(EOS_TOKEN)
                                _touch_progress()
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # Primary path: OpenAI chat.completion.chunk streaming shape
                        token_text = ""
                        if isinstance(obj.get("choices"), list) and obj["choices"]:
                            choice = obj["choices"][0]
                            if isinstance(choice, dict):
                                delta = choice.get("delta") or {}
                                # Ignore role-only deltas
                                token_text = delta.get("content") or ""
                                # If a finish_reason is present without new text, consider completion done
                                if not token_text and choice.get("finish_reason"):
                                    with _buffer_lock:
                                        _token_buffer.append(EOS_TOKEN)
                                        _touch_progress()
                                    break
                        # Fallbacks for non-standard responses from some llama.cpp variants
                        if not token_text:
                            # Legacy /completion shapes
                            if isinstance(obj.get("token"), dict):
                                token_text = obj["token"].get("text", "")
                            elif "content" in obj:
                                token_text = obj["content"]
                            elif "completion" in obj:
                                token_text = obj["completion"]

                            # If the variant returns a stop/done flag
                            if obj.get("stop") or obj.get("done"):
                                with _buffer_lock:
                                    _token_buffer.append(EOS_TOKEN)
                                    _touch_progress()
                                break

                        if token_text:
                            accum_text += token_text
                            toks = encoding.encode(accum_text, allowed_special="all")
                            if len(toks) > last_len:
                                new_toks = toks[last_len:]
                                with _buffer_lock:
                                    _token_buffer.extend(new_toks)
                                    _touch_progress()
                                last_len = len(toks)
                _stream_done.set()
            except Exception as e:  # pragma: no cover - network errors
                _stream_error = e
                _stream_done.set()

        t = threading.Thread(target=run, name="llama-server-stream", daemon=True)
        t.start()
        return t

    def infer_next_token(
        tokens: list[int], temperature: float = 0.0, new_request: bool = False
    ) -> int:
        global _stream_thread
        if new_request:
            _reset_stream_state()
            _stream_thread = _start_stream(tokens, temperature)
            start = _now()
            while _now() - start < FIRST_BYTE_TIMEOUT_S:
                with _buffer_lock:
                    if _token_buffer:
                        tok = _token_buffer.pop(0)
                        _touch_progress()
                        return tok
                if _stream_error is not None:
                    raise RuntimeError(f"llama-server stream error: {_stream_error!r}")
                time.sleep(POLL_INTERVAL_S)
            return EOS_TOKEN

        if _stream_error is not None:
            raise RuntimeError(f"llama-server stream error: {_stream_error!r}")

        wait_start = _now()
        while _now() - wait_start < CALL_MAX_WAIT_S:
            with _buffer_lock:
                if _token_buffer:
                    tok = _token_buffer.pop(0)
                    _touch_progress()
                    return tok
            if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
                return EOS_TOKEN
            time.sleep(POLL_INTERVAL_S)

        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        time.sleep(POLL_INTERVAL_S)
        with _buffer_lock:
            if _token_buffer:
                tok = _token_buffer.pop(0)
                _touch_progress()
                return tok

        if _now() - _last_progress_ts > NO_TOKEN_TIMEOUT_S:
            return EOS_TOKEN

        return 0

    return infer_next_token
