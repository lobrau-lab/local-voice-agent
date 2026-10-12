"""Agent construction, audio I/O, speech-to-text, and text-to-speech."""

import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    CHANNELS,
    SILENCE_THRESHOLD,
    SILENCE_DURATION_SEC,
    MAX_RECORD_SEC,
    MIN_RECORD_SEC,
    WHISPER_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    TTS_LANG_CODE,
    MLX_SERVER_BASE_URL,
    MLX_SERVER_MODEL,
    MAX_TOOL_ITERATIONS,
    NOISE_MULTIPLIER,
    MIN_NOISE_FLOOR,
)
from tools import ALL_TOOLS

# ═══════════════════════════════════════════════════════════════════════
#  Noise calibration
# ═══════════════════════════════════════════════════════════════════════


def calibrate_noise(
    duration: float = 1.5,
    noise_multiplier: float = NOISE_MULTIPLIER,
    min_noise_floor: float = MIN_NOISE_FLOOR,
) -> float:
    """Sample ambient noise and return a silence threshold.

    The threshold is `noise_multiplier` x the ambient RMS, with a floor of `min_noise_floor`,
    so it adapts to noisy or quiet environments automatically.
    """
    print(f"\U0001f39a\ufe0f  Calibrating... stay quiet for {duration:.1f}s")
    samples = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    ambient = float(np.sqrt(np.mean(samples**2)))
    threshold = max(ambient * noise_multiplier, min_noise_floor)
    print(f"   Ambient RMS: {ambient:.4f}  ->  threshold: {threshold:.4f}")
    return threshold


# ═══════════════════════════════════════════════════════════════════════
#  Audio recording
# ═══════════════════════════════════════════════════════════════════════


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float | None = None,
    silence_duration: float = SILENCE_DURATION_SEC,
    max_duration: float = MAX_RECORD_SEC,
    min_duration: float = MIN_RECORD_SEC,
    noise_multiplier: float = NOISE_MULTIPLIER,
    min_noise_floor: float = MIN_NOISE_FLOOR,
) -> np.ndarray:
    """Record from the mic until the user stops speaking.

    If silence_threshold is None, auto-calibrates from ambient noise.
    Returns float32 audio samples.
    """
    if silence_threshold is None:
        silence_threshold = calibrate_noise(
            noise_multiplier=noise_multiplier, min_noise_floor=min_noise_floor
        )

    q: "queue.Queue[np.ndarray]" = queue.Queue()
    blocks: list[np.ndarray] = []
    silence_start: float | None = None
    start_time = time.time()
    stopped = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=int(sample_rate * 0.05),
    )

    with stream:
        print("\U0001f3a4 Listening... (speak, then pause)")
        while not stopped.is_set():
            try:
                block = q.get(timeout=0.1)
            except queue.Empty:
                continue

            blocks.append(block)
            elapsed = time.time() - start_time
            rms = float(np.sqrt(np.mean(block**2)))

            if rms < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    if elapsed >= min_duration:
                        stopped.set()
                    else:
                        silence_start = None
            else:
                silence_start = None

            if elapsed >= max_duration:
                print(f"Max recording length ({max_duration}s) reached.")
                stopped.set()

    if not blocks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(blocks, axis=0).flatten()
    print(f"Recorded {len(audio) / sample_rate:.1f}s of audio.")
    return audio


# ═══════════════════════════════════════════════════════════════════════
#  Speech-to-text
# ═══════════════════════════════════════════════════════════════════════


def transcribe_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe float32 audio using mlx-whisper."""
    import mlx_whisper

    if len(audio) == 0:
        return ""

    if sample_rate != 16000:
        ratio = 16000 / sample_rate
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio), new_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_MODEL,
        fp16=True,
        verbose=False,
    )
    return (result.get("text") or "").strip()


# ═══════════════════════════════════════════════════════════════════════
#  Text-to-speech
# ═══════════════════════════════════════════════════════════════════════

_tts_model = None


def _get_tts_model():
    global _tts_model
    if _tts_model is None:
        from mlx_audio.tts.utils import load_model

        print("Loading TTS model (first time only)...")
        _tts_model = load_model(TTS_MODEL)
    return _tts_model


def _synthesise(text: str) -> tuple[np.ndarray, int]:
    """Generate TTS audio. Returns (float32 samples, sample_rate)."""
    model = _get_tts_model()

    chunks: list[np.ndarray] = []
    sr: int | None = None

    try:
        for result in model.generate(
            text,
            voice=TTS_VOICE,
            lang_code=TTS_LANG_CODE,
        ):
            audio = getattr(result, "audio", None)
            if audio is None:
                audio = result
            arr = np.asarray(audio).astype(np.float32).flatten()
            chunks.append(arr)
            if sr is None:
                sr = int(getattr(result, "sample_rate", 0)) or None
    except TypeError:
        for result in model.generate(text, voice=TTS_VOICE):
            audio = getattr(result, "audio", None)
            if audio is None:
                audio = result
            arr = np.asarray(audio).astype(np.float32).flatten()
            chunks.append(arr)
            if sr is None:
                sr = int(getattr(result, "sample_rate", 0)) or None

    if not chunks:
        return np.array([], dtype=np.float32), sr or 24000

    return np.concatenate(chunks), sr or int(getattr(model, "sample_rate", 24000))


def speak(text: str) -> None:
    """Synthesise and play speech."""
    if not text.strip():
        return

    print(f"\U0001f50a Speaking: {text[:120]}{'...' if len(text) > 120 else ''}")
    try:
        audio, sr = _synthesise(text)
    except Exception as e:
        print(f"[TTS ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return

    if len(audio) == 0:
        return

    sd.play(audio, sr)
    sd.wait()


# ═══════════════════════════════════════════════════════════════════════
#  Agent
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a highly capable, fully agentic, concise voice assistant running locally on "
    "the user's Mac. You have full file-system, shell, web, and API access via tools.\n\n"
    "CRITICAL RULES - follow these exactly:\n\n"
    "1. You have the highest standards of agentic capabilities. Anticipate that users may be vague or incomplete in their prompts. Instead of immediately asking for clarification, use your tools (like web search, shell, file search) to actively figure out the missing context, find the right paths, or solve the problem autonomously first.\n"
    "2. Keep spoken replies short and natural; this is a voice conversation.\n"
    "3. ALWAYS CALL THE TOOL. Never say 'I cannot' or 'I don't have access' without first calling the relevant tool.\n"
    "4. PATH CORRECTION: If the user's path may not be exact, DO NOT ask them. Instead, use shell tools (like find, ls) to locate the correct file or directory on their behalf.\n"
    "5. Chain multiple tool calls when needed to accomplish complex tasks.\n"
    "6. REPORT FAITHFULLY. When a tool returns [ERROR], say exactly what the error was.\n"
    "7. Never output raw JSON or tool-call syntax; speak naturally.\n"
    "8. Use save_memory when the user asks you to remember something.\n"
    "9. Use read_memory at the start of a session to recall context.\n"
    "10. Ask for confirmation ONLY before doing something highly destructive (e.g. rm -rf, dropping a DB) or extremely slow.\n"
)


def build_agent():
    """Create the agent_weave Agent pointing at the local MLX server."""
    from agent_weave import Agent
    from agent_weave.llm.openai_backend import OpenAIBackend
    from agent_weave.memory import ConversationMemory

    backend = OpenAIBackend(
        api_key="local",
        base_url=MLX_SERVER_BASE_URL,
        default_model=MLX_SERVER_MODEL,
    )

    agent = Agent(
        name="local-voice-agent",
        llm=backend,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        memory=ConversationMemory(),
        model=MLX_SERVER_MODEL,
        max_iterations=MAX_TOOL_ITERATIONS,
        verbose=True,
    )
    return agent
