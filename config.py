"""Central configuration for the local voice agent."""

import os

# ── Paths ──────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
AGENT_DIR = os.path.join(HOME, ".my_voice_agent")
MEMORY_DIR = os.path.join(AGENT_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ── MLX Server ─────────────────────────────────────────────────────────
MLX_SERVER_HOST = "127.0.0.1"
MLX_SERVER_PORT = 8080
MLX_SERVER_BASE_URL = f"http://{MLX_SERVER_HOST}:{MLX_SERVER_PORT}/v1"
MLX_SERVER_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# ── Whisper (Speech-to-Text) ───────────────────────────────────────────
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

# ── TTS ────────────────────────────────────────────────────────────────
# Kitten TTS or Kokoro work well with mlx-audio 0.5.x
TTS_MODEL = "mlx-community/Kokoro-82M-4bit"
TTS_VOICE = "af_heart"           # Kokoro voice id
TTS_LANG_CODE = "a"              # Kokoro lang code ("a" = American English)

# ── Audio Recording ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION_SEC = 1.5
MAX_RECORD_SEC = 30.0
MIN_RECORD_SEC = 1.0

# ── Agent Behaviour ────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS = 10
