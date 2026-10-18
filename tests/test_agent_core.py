import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import time

import agent_core

# -----------------------------------------------------------------------------
# Test calibrate_noise
# -----------------------------------------------------------------------------


@patch("agent_core.sd.wait")
@patch("agent_core.sd.rec")
def test_calibrate_noise(mock_rec, mock_wait):
    mock_samples = np.full((100, 1), 0.1, dtype="float32")
    mock_rec.return_value = mock_samples

    threshold = agent_core.calibrate_noise(
        duration=0.5, noise_multiplier=3.0, min_noise_floor=0.05
    )

    mock_rec.assert_called_once()
    mock_wait.assert_called_once()
    assert np.isclose(threshold, 0.3)


@patch("agent_core.sd.wait")
@patch("agent_core.sd.rec")
def test_calibrate_noise_min_floor(mock_rec, mock_wait):
    mock_samples = np.full((100, 1), 0.001, dtype="float32")
    mock_rec.return_value = mock_samples

    threshold = agent_core.calibrate_noise(
        duration=0.5, noise_multiplier=3.0, min_noise_floor=0.05
    )

    assert threshold == 0.05


# -----------------------------------------------------------------------------
# Test record_until_silence
# -----------------------------------------------------------------------------


@patch("agent_core.sd.InputStream")
def test_record_until_silence_stops_on_silence(mock_input_stream):
    import threading

    callback_ref = [None]

    def mock_init(*args, **kwargs):
        callback_ref[0] = kwargs.get("callback")
        return MagicMock()

    mock_input_stream.side_effect = mock_init

    def simulate_audio():
        time.sleep(0.1)
        cb = callback_ref[0]
        if not cb:
            return

        noise_block = np.full((1000, 1), 0.2, dtype="float32")
        cb(noise_block, len(noise_block), None, None)

        silence_block = np.full((1000, 1), 0.01, dtype="float32")
        cb(silence_block, len(silence_block), None, None)
        time.sleep(0.6)
        cb(silence_block, len(silence_block), None, None)

    t = threading.Thread(target=simulate_audio)
    t.start()

    audio = agent_core.record_until_silence(
        silence_threshold=0.1, silence_duration=0.5, min_duration=0.1, max_duration=5.0
    )
    t.join()

    assert len(audio) > 0
    assert audio.dtype == np.float32


@patch("agent_core.sd.InputStream")
def test_record_until_silence_max_duration(mock_input_stream):
    import threading

    callback_ref = [None]

    def mock_init(*args, **kwargs):
        callback_ref[0] = kwargs.get("callback")
        return MagicMock()

    mock_input_stream.side_effect = mock_init

    def simulate_audio():
        time.sleep(0.1)
        cb = callback_ref[0]
        if not cb:
            return

        noise_block = np.full((1000, 1), 0.2, dtype="float32")
        cb(noise_block, len(noise_block), None, None)
        time.sleep(0.6)
        cb(noise_block, len(noise_block), None, None)

    t = threading.Thread(target=simulate_audio)
    t.start()

    audio = agent_core.record_until_silence(
        silence_threshold=0.1, silence_duration=1.0, min_duration=0.1, max_duration=0.5
    )
    t.join()

    assert len(audio) > 0


# -----------------------------------------------------------------------------
# Test transcribe_audio
# -----------------------------------------------------------------------------


@patch.dict("sys.modules", {"mlx_whisper": MagicMock()})
def test_transcribe_audio_empty():
    assert agent_core.transcribe_audio(np.array([])) == ""


@patch.dict("sys.modules")
def test_transcribe_audio_resample():
    import sys

    mlx_whisper_mock = MagicMock()
    mlx_whisper_mock.transcribe.return_value = {"text": "hello"}
    sys.modules["mlx_whisper"] = mlx_whisper_mock

    audio = np.ones(8000, dtype=np.float32)
    result = agent_core.transcribe_audio(audio, sample_rate=8000)

    assert result == "hello"

    call_args = mlx_whisper_mock.transcribe.call_args
    assert len(call_args[0][0]) == 16000


# -----------------------------------------------------------------------------
# Test TTS
# -----------------------------------------------------------------------------


@patch("agent_core._get_tts_model")
def test_synthesise(mock_get_model):
    mock_model = MagicMock()

    class DummyResult:
        def __init__(self):
            self.audio = np.array([1, 2, 3], dtype=np.float32)
            self.sample_rate = 24000

    mock_model.generate.return_value = [DummyResult()]
    mock_get_model.return_value = mock_model

    audio, sr = agent_core._synthesise("test text")

    assert sr == 24000
    assert len(audio) == 3
    assert np.array_equal(audio, np.array([1, 2, 3], dtype=np.float32))


@patch("agent_core._synthesise")
@patch("agent_core.sd.play")
@patch("agent_core.sd.wait")
def test_speak(mock_wait, mock_play, mock_synth):
    mock_synth.return_value = (np.array([1, 2, 3], dtype=np.float32), 24000)

    agent_core.speak("Hello world")

    mock_synth.assert_called_once_with("Hello world")
    mock_play.assert_called_once()
    mock_wait.assert_called_once()


@patch("agent_core.sd.play")
def test_speak_empty(mock_play):
    agent_core.speak("")
    agent_core.speak("   ")
    mock_play.assert_not_called()


# -----------------------------------------------------------------------------
# Test build_agent
# -----------------------------------------------------------------------------


@patch("agent_weave.Agent")
@patch("agent_weave.memory.ConversationMemory")
@patch("agent_weave.llm.openai_backend.OpenAIBackend")
def test_build_agent(mock_backend, mock_memory, mock_agent):
    mock_backend_instance = MagicMock()
    mock_backend.return_value = mock_backend_instance

    mock_memory_instance = MagicMock()
    mock_memory.return_value = mock_memory_instance

    agent = agent_core.build_agent()

    mock_backend.assert_called_once()
    mock_memory.assert_called_once()
    mock_agent.assert_called_once()

    kwargs = mock_agent.call_args.kwargs
    assert kwargs["llm"] == mock_backend_instance
    assert kwargs["memory"] == mock_memory_instance
    assert kwargs["name"] == "local-voice-agent"
