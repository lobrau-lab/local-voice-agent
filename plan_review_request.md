# Plan Review Request

**Task**: Write a test file (`tests/test_agent_core.py`) for `agent_core.py` to increase testing reliability and coverage, by mocking hardware (microphone/speaker) and ML models (Whisper/TTS).

**Analysis of `agent_core.py`**:
* `calibrate_noise()`: Reads from `sounddevice.rec()` and uses `sounddevice.wait()`, returns a threshold.
* `record_until_silence()`: Uses `sounddevice.InputStream` and queues/callbacks to read mic data until silence. Uses `calibrate_noise()` if threshold not passed.
* `transcribe_audio()`: Loads `mlx_whisper` and transcribes an `np.ndarray`.
* `_get_tts_model()`: Loads an `mlx_audio.tts` model and caches it.
* `_synthesise()`: Takes text and uses the model to generate audio (returns an array and a sample rate).
* `speak()`: Calls `_synthesise()` and then `sounddevice.play()`.
* `build_agent()`: Creates an `Agent` via `agent_weave` targeting a local OpenAIBackend endpoint.

**Test Strategy (`pytest` framework)**:
Create `tests/test_agent_core.py`. We will mock out `sounddevice` and ML models so tests can run without actual hardware or ML models downloading/running.

1. **Test `calibrate_noise`**:
   - Mock `sd.rec` and `sd.wait`. `sd.rec` returns mocked silence/noise array.
   - Assert `sd.rec` is called correctly and threshold is calculated as expected.

2. **Test `record_until_silence`**:
   - Mock `sd.InputStream`. We need to simulate the callback feeding data into the queue.
   - We can mock the `InputStream` context manager (`__enter__`, `__exit__`), and directly put mocked noise followed by silence into the queue, or just patch `time.time()` or directly simulate the queue `get` within the loop to avoid real sleeping.
   - Verify it handles early stopping (silence triggers stop) or max duration.

3. **Test `transcribe_audio`**:
   - Mock `mlx_whisper`.
   - Call `transcribe_audio` with a dummy `np.ndarray`.
   - Ensure it returns the mocked text.
   - Ensure it resamples if `sample_rate != 16000`.

4. **Test Text-to-Speech (`_get_tts_model`, `_synthesise`, `speak`)**:
   - Mock `mlx_audio.tts.utils.load_model`.
   - Mock the TTS model's `generate` method to return dummy audio objects.
   - For `speak`, mock `sd.play` and `sd.wait`.
   - Verify `speak` doesn't do anything on empty string.
   - Verify `speak` calls `sd.play` with the array returned by `_synthesise`.

5. **Test `build_agent`**:
   - Mock `agent_weave.Agent`, `agent_weave.llm.openai_backend.OpenAIBackend`, and `agent_weave.memory.ConversationMemory` to avoid network/disk setup.
   - Check if agent is instantiated with right params from `config`.

**Plan Details**:
1. Implement `tests/test_agent_core.py`.
2. Add comprehensive mocks using `unittest.mock.patch` for `sounddevice`, `mlx_whisper`, `mlx_audio`, and `time` where necessary.
3. Test edge cases (e.g. empty audio array, silence duration logic, resampling logic).
4. Run `pytest tests/test_agent_core.py` to ensure it passes.
5. Format code with `black`.
6. Complete pre commit steps.
7. Submit the PR.

Is this plan solid? Any improvements on the mocking strategy for the audio stream callback?
