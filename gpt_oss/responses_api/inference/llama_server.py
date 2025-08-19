"""Inference backend using a local `llama.cpp` server.

This backend connects to a running `llama.cpp` HTTP server (the
`llama-server` binary) and streams generated tokens.  The server is
expected to expose the `/completion` endpoint with streaming enabled.
"""

import json
import threading
import time
from typing import Callable, Optional

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

    def _start_stream(token_ids: list[int], temperature: float) -> threading.Thread:
        prompt_text = encoding.decode(token_ids)

        def run() -> None:
            nonlocal prompt_text, temperature
            global _stream_error
            accum_text = ""
            last_len = 0
            try:
                url = "http://localhost:8080/completion"
                payload = {
                    "prompt": prompt_text,
                    "temperature": temperature,
                    "stream": True,
                }
                with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                        else:
                            data = line.strip()
                        if data == "[DONE]":
                            with _buffer_lock:
                                _token_buffer.append(EOS_TOKEN)
                                _touch_progress()
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("stop"):
                            with _buffer_lock:
                                _token_buffer.append(EOS_TOKEN)
                                _touch_progress()
                            break
                        token_text = ""
                        if isinstance(obj.get("token"), dict):
                            token_text = obj["token"].get("text", "")
                        elif "content" in obj:
                            token_text = obj["content"]
                        elif "completion" in obj:
                            token_text = obj["completion"]
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
