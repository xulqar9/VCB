# VCB Deep Technical Audit

**Audit date:** 2026-09-21  
**Scope:** Python voice pipeline, memory, interruption, concurrency, LLM/STT/TTS, avatar bridge, lifecycle, and tests.  
**Method:** Source inspection of the current workspace. No project files were modified during the audit.

`[VERIFIED]` means directly confirmed in the current source. `[INFERRED]` means deduced from the implementation but not runtime-tested here. `[UNVERIFIED]` means the workspace did not provide enough evidence to confirm it.

## Executive Summary

VCB has a mature asyncio-based streaming architecture. The user-visible interruption path is substantially correct: it cancels the active turn, stops local playback, discards stale queued audio, interrupts the avatar, and cancels the OpenAI-compatible LLM stream. The strongest parts are the playback generation counter, device supervisors, Windows Job Object cleanup, cross-thread SQLite locking, and avatar control/audio queue separation.

The most important weaknesses are:

1. Mid-stream LLM retries can duplicate already-spoken output.
2. STT and the active TTS synthesis run in `asyncio.to_thread`; cancelling the awaiting task does not stop the underlying worker or GPU operation.
3. There is effectively no automated test suite in the current workspace.
4. Shutdown and background consolidation have edge cases around cancellation and closing the LLM.
5. Runtime state is represented by several interacting booleans instead of an explicit state machine.

## 1. Architecture and Component Interaction

### Implemented

[VERIFIED] [orchestrator.py](vcb/orchestrator.py) coordinates:

```text
microphone callback
	-> MicCapture queue
	-> _mic_loop
	-> Silero VAD
	-> SpeechStart / SpeechEnd
	-> _event_loop
	-> turn task: STT -> memory -> LLM -> chunker -> TTS queue
	-> _tts_loop
	-> SpeakerPlayback and/or AvatarBridge
```

The LLM, STT, and TTS engines are injected behind abstract interfaces. Audio callbacks run on PortAudio threads and marshal work to asyncio with `call_soon_threadsafe`. CPU/GPU model calls are generally moved to worker threads with `asyncio.to_thread`.

Optional subsystems include long-term memory, background consolidation, JSON-protocol skills, proactive AI-led turns, and an Unreal avatar bridge.

### Strengths

- Clear ownership boundaries between capture, VAD, turn processing, synthesis, playback, and avatar transport.
- Engine interfaces make STT, LLM, and TTS implementations replaceable.
- The event loop is protected from blocking PortAudio callbacks and most model inference.
- Queue backpressure and generation checks prevent a fast producer from indefinitely overwhelming playback.
- Configuration is centralized in typed dataclasses.

### Problems and missing pieces

[VERIFIED] `VADSegmenter` loads Silero with `torch.hub.load()` in its constructor ([vad.py](vcb/vad.py)), rather than in an async model-loading phase. This can block construction and potentially perform network/disk work on the event-loop thread.

[INFERRED] `_event_loop` has no outer error boundary. An unexpected exception from a future avatar or task-management change could terminate the event loop and cause `run()` to shut down the session.

[UNVERIFIED] Full end-to-end operation with the active real model, audio devices, and Unreal client was not run as part of this audit.

### Best-practice assessment

The broad architecture follows current asyncio voice-pipeline practices. Model loading and task supervision should be made more consistent before treating it as production-grade.

## 2. Memory and Consolidation

### Implemented

[VERIFIED] `MemoryStore` in [store.py](vcb/memory/store.py) provides:

- SQLite persistence for turns, facts, summaries, and notes.
- Ephemeral in-memory mode.
- CPU embeddings and normalized vector similarity recall.
- Recent-history exclusion.
- Fact upserts and rolling summaries.
- Thread-safe database access with `check_same_thread=False` and a lock.

[VERIFIED] `Orchestrator._remember_user()` embeds the query, recalls facts/summary/relevant turns, stores the user turn, and injects a bounded memory block into the next prompt.

[VERIFIED] `_remember_assistant()` stores completed assistant replies. Consolidation runs in a background task after a configurable number of turns, and a final consolidation is attempted during shutdown.

### Strengths

- SQLite access is correctly serialized for `to_thread` workers.
- Query-side BGE instruction handling is appropriate for asymmetric retrieval.
- Memory failures are isolated from the conversation path and logged.
- Consolidation parse failures return no data rather than corrupting the store.
- Fact updates use an atomic SQLite upsert.

### Problems and missing pieces

[VERIFIED] Interrupted assistant partials are added to live conversation history but are not passed to `_remember_assistant()`. They are therefore absent from long-term memory.

[VERIFIED] The in-memory embedding matrix grows indefinitely and `np.vstack()` copies the full matrix on every inserted embedding. This is acceptable for small stores but scales poorly for long-lived use.

[VERIFIED] Shutdown waits up to 20 seconds for an existing consolidation task, then can proceed to `llm.close()` while that task is still running if the timeout expires. The task may then fail against a closed LLM client.

[INFERRED] Memory injection can duplicate information already present in recent conversation history because recalled turns are only excluded by count, not by semantic or session-aware deduplication.

### Best-practice assessment

The locking and failure isolation are solid. A production memory system would need bounded/indexed retrieval, explicit shutdown cancellation, and a deliberate policy for interrupted partials.

## 3. Interruption and Barge-In

### Implemented

[VERIFIED] On `SpeechStart`, the event loop interrupts when playback, a turn, or avatar speech is active. `_cancel_turn()`:

1. Marks the active metrics as interrupted.
2. Cancels and awaits the turn task.
3. Drains the TTS text queue.
4. Flushes local playback.
5. Cancels the avatar end task.
6. Clears queued avatar audio and sends an avatar interrupt frame.

[VERIFIED] The LLM cancellation path propagates `CancelledError`, and the OpenAI-compatible stream exits without retrying cancellation. Closing the HTTP stream should stop server-side generation.

[VERIFIED] `SpeakerPlayback.flush()` increments a generation counter. `_tts_loop` and the playback pump reject chunks from older generations, preventing stale audio from reaching speakers after a barge-in.

[VERIFIED] Interrupted spoken text is saved in live conversation history and `_resume_pending` causes the next model turn to receive a resume instruction.

### What is actually cancelled

| Pipeline stage | Result | Evidence |
|---|---|---|
| LLM HTTP stream | Cancelled | `OpenAICompatLLM.stream()` re-raises `CancelledError` |
| Local queued/output audio | Cancelled | playback generation and flush logic |
| Avatar queued audio | Cancelled | `AvatarBridge.interrupt()` drains audio queue |
| Active TTS synthesis | Not truly cancelled | current engines use `to_thread`; worker continues |
| Active STT transcription | Not truly cancelled | faster-whisper runs in `to_thread` |
| Active embedding/database worker | Not truly cancelled | same `to_thread` behavior |

The interruption therefore does more than merely stop audio, but it does not cancel every underlying computation. The current TTS result is discarded after the worker returns, and an in-flight STT worker continues after its awaiter is cancelled.

### Strengths

- The generation counter is a strong and correct stale-output defense.
- The turn task is awaited after cancellation, preventing intentional overlapping turn tasks.
- Avatar playback-clock cancellation prevents stale `speech_end` events after interruption.
- Resume behavior avoids a second LLM request and lets the model decide whether to continue or answer a new topic.

### Problems and missing pieces

[INFERRED] Rapid interruptions can leave several detached STT/TTS/GPU worker operations competing for the thread pool and GPU. This can increase latency for the next real turn.

[VERIFIED] The active TTS engine's default `synthesize_stream()` is one-shot, so a whole sentence may be generated before cancellation is observed by the asyncio task.

[VERIFIED] Interrupted assistant partials are not persisted to long-term memory.

### Best-practice assessment

The user-visible path is strong. True cancellation of GPU inference requires engine-level cooperative cancellation or smaller generation units; Python task cancellation alone cannot stop arbitrary `to_thread` work.

## 4. Async and Concurrency

### Strengths

- Shared state is mutated on one asyncio event loop, avoiding ordinary data races.
- PortAudio callbacks communicate through thread-safe loop handoff.
- SQLite mutations are explicitly locked.
- Background consolidation has an overlap guard.
- Playback and avatar transport use bounded queues where real-time behavior matters.

### Problems

[VERIFIED] `self._metrics` is a shared mutable field. `_tts_loop` reads whichever metrics object is current, so audio belonging to an old turn can be attributed to a newer turn after a barge-in.

[VERIFIED] `_tts_queue.put()` can block `_stream_reply()` when TTS is slower than LLM generation. This is valid backpressure, but it keeps the LLM request alive and occupies a server slot while waiting for synthesis.

[INFERRED] The avatar `_wake` event has a subtle check/clear race pattern. It is unlikely to lose work permanently because queue checks repeat, but a condition-variable-style queue or a single producer-consumer notification primitive would be easier to reason about.

[VERIFIED] Consolidation shares the same LLM client with live turns. The server is configured for multiple slots, but this behavior is dependent on server capacity and has not been proven with every active model preset.

### Best-practice assessment

The design is generally sound for a single-loop real-time application. Explicit per-turn context and structured task supervision would reduce the remaining concurrency risk.

## 5. LLM Pipeline and Tool Loop

### Implemented

[VERIFIED] [openai_compat.py](vcb/llm/openai_compat.py) streams from an OpenAI-compatible endpoint, supports per-request temperature/token overrides, and retries connection/timeouts.

[VERIFIED] The orchestrator decides early whether a response is a JSON skill call or spoken text, streams spoken chunks to TTS, parses late tool JSON, dispatches skills, appends the tool result, and asks the model for a natural spoken answer.

### Strengths

- Cancellation is explicitly propagated.
- Tool calls are not intentionally voiced.
- Tool iteration has a configurable cap.
- Tool dispatch is isolated behind a registry and returns error text instead of raising through the main turn.
- Prompt/history handling is simple and inspectable.

### Verified bug

[VERIFIED] Mid-stream retry can duplicate speech. The retry loop surrounds the entire response iteration. If the connection fails after some deltas have already been yielded, the client starts a fresh request and yields the beginning again. Already-spoken content can therefore be emitted twice.

The retry policy should distinguish pre-stream connection failure from failure after the first yielded delta. Once output has been yielded, the failure should normally propagate or use an explicit continuation strategy; restarting the full request is unsafe.

### Other concerns

[VERIFIED] A streaming failure after some spoken chunks is caught by `_speak_reply`; because `parts` is non-empty, no apology is queued and the partial reply is persisted as a completed assistant message. This may be intentional, but it conflates a failed response with a successfully completed response.

[INFERRED] Tool detection is heuristic and depends on model formatting. The tolerant parser is pragmatic, but live adherence with every model preset remains unverified.

## 6. STT and VAD

### Implemented

[VERIFIED] Silero VAD segments 32 ms frames, provides pre-roll, emits early `SpeechStart`, trims trailing silence, and enforces a maximum utterance length.

[VERIFIED] faster-whisper runs in a worker thread, uses the already-segmented audio without its own VAD, and uses greedy decoding for latency.

### Strengths

- Early start events support barge-in.
- Pre-roll helps preserve initial phonemes.
- Mic queue overflow drops oldest frames, preserving current live audio.
- Mic supervisor reopens failed devices with backoff.

### Problems and missing pieces

[VERIFIED] VAD model loading occurs during object construction rather than through the asynchronous startup lifecycle.

[INFERRED] A cancelled STT task does not stop faster-whisper inference already running in its worker thread.

[UNVERIFIED] No automated coverage verifies VAD transitions, false starts, max-duration flushes, or audio trimming.

## 7. TTS and Playback

### Implemented

[VERIFIED] TTS engines expose async load/synthesize/close. The default base streaming method wraps one-shot synthesis; engines may override it.

[VERIFIED] `_tts_loop` handles text chunks, tags, gain limiting, local playback, avatar forwarding, prosody pauses, stale-generation checks, and TTS cleanup with `agen.aclose()`.

[VERIFIED] `SpeakerPlayback` uses a PortAudio callback, an asyncio pump, a jitter/pre-roll buffer, generation-based flushing, and device recovery.

### Strengths

- The playback generation design handles stale queued audio well.
- Audio callbacks do only lightweight buffer slicing.
- Device failures are supervised and reopened.
- The pre-roll buffer reduces streamed TTS underruns.
- TTS failures do not kill the full session.

### Problems

[VERIFIED] One-shot TTS generation limits cancellation responsiveness and wastes compute after interruption.

[INFERRED] A TTS engine failure after earlier chunks may leave a partial assistant response that is treated as complete, as described above.

[UNVERIFIED] Engine-specific close behavior and GPU memory release were not runtime-tested in this audit.

## 8. State Management

### Implemented

[VERIFIED] State is represented by `_turn_task`, `_resume_pending`, `_user_speaking`, `_avatar_speaking`, `_last_activity`, avatar playback timing, consolidation flags, and speaker state.

### Strengths

- `_turn_active()` derives activity from the task rather than maintaining another independent flag.
- Event-loop serialization prevents simultaneous mutation of these fields.
- Avatar state timing correctly distinguishes synthesis completion from playback completion.

### Problems

[VERIFIED] Several guards independently combine `_turn_active()`, `speaker.is_playing`, `_user_speaking`, and `_avatar_speaking`. This implicit state machine is difficult to extend and has a high regression risk.

[VERIFIED] Conversation trimming uses `self._history[-max_msgs:]`. It does not explicitly preserve user/assistant pairs or tool-related message relationships, although tool messages are currently not persisted in normal history.

### Improvement

Introduce a small explicit state model for listening, thinking, speaking, interrupted, and idle. Keep the existing playback and avatar mechanisms, but derive guards and transitions from one source of truth.

## 9. Error Handling and Resource Cleanup

### Strengths

- Device supervisors use exponential backoff.
- Turn failures are logged without ending the session.
- Engine cleanup is gathered with `return_exceptions=True`.
- Windows Job Object cleanup protects child processes if Python is hard-killed.
- Explicit server shutdown remains the primary cleanup path.

### Problems

[VERIFIED] `shutdown()` can be called after partial startup. `run()` uses `asyncio.gather()` for model loading, so one load failure can leave other models partially initialized. Teardown is mostly guarded but has not been designed as a fully idempotent lifecycle state machine.

[VERIFIED] Consolidation timeout can leave a task running while the LLM is closed.

[VERIFIED] `ManagedProcesses._assign_to_job()` accesses private `Popen._handle`. This is pragmatic on Windows but less stable than a documented process-handle API.

[VERIFIED] Some device close exceptions are swallowed completely, which helps teardown but can hide diagnostic information.

## 10. Avatar Bridge

### Implemented

[VERIFIED] [bridge.py](vcb/avatar/bridge.py) provides a local framed TCP server with separate control and bounded audio queues. Control messages have priority and are never dropped; audio drops oldest frames under pressure. Audio is split into approximately 20 ms frames. `speech_flush` intentionally travels through the audio queue so it follows queued audio.

### Strengths

- The control/audio queue separation is exactly the right policy for state correctness.
- Barge-in drains stale audio before sending interrupt.
- Single-client replacement is simple and appropriate for the current local integration.
- The bridge does not block the voice pipeline on a slow avatar client.

### Concerns

[INFERRED] The event wake-up logic is more subtle than necessary and could be replaced with a queue-driven sender pattern that avoids manual event clearing.

[UNVERIFIED] Reconnect behavior during an active utterance and sustained slow-client backpressure were not stress-tested here.

## 11. Tests and Coverage

[VERIFIED] The current workspace contains no normal `tests/` suite. `tools/test_action.py` is a manual avatar action harness, not an automated test. The current source tree also contains no `pytest`/`unittest` test functions.

High-value missing tests:

1. Playback generation and stale-audio rejection after interruption.
2. `_cancel_turn()` partial-history and resume behavior.
3. Mid-stream LLM failure to prove no duplicate speech.
4. Chunker first-clause release, action-span holding, hard limits, and flush.
5. Sanitizer handling of tags, leaked JSON, speaker labels, split directives, and punctuation.
6. Memory recall thresholds, recent exclusion, persistence, facts, consolidation ranges, and concurrent access.
7. Skill parser tolerance and calculator safety.
8. Avatar queue priority, audio drop behavior, interrupt drain, and reconnects.
9. VAD start/end transitions, pre-roll, trailing trim, and maximum utterance handling.
10. Lifecycle tests for partial startup and shutdown timeout behavior.

Most of these can run without GPU hardware by using fake LLM/STT/TTS implementations.

## Prioritized Improvements

### Priority 1: correctness -- DONE

1. Fix mid-stream LLM retry so a request is not restarted after deltas have been emitted.
2. Add tests for interruption, generation counters, sanitizer behavior, and retry failure.
3. Make turn completion distinguish successful replies from failed-but-partial replies.

### Priority 2: cancellation and lifecycle -- DONE

4. Add cooperative cancellation/supersession to STT and TTS engines where supported.
5. Cancel or finish timed-out consolidation before closing the LLM.
6. Make startup/shutdown explicitly idempotent and safe after partial initialization.
7. Move VAD model loading into the asynchronous startup path.

### Priority 3: maintainability and scale -- DONE

8. Replace shared `_metrics` with per-turn context captured by the TTS work item.
   *Each queued `_TTSItem` now carries its turn's `TurnMetrics`; the TTS loop no longer reads the shared field.*
9. Introduce an explicit conversation/audio state machine.
   *[state.py](vcb/state.py): `PipelineSnapshot` derives one `Phase` (idle/listening/thinking/speaking) plus the `assistant_active` / `busy` guards from the existing signals; barge-in, idle-loop, and lead-turn guards all use it.*
10. Bound or index long-term embedding storage and avoid repeated full `vstack()` copies.
    *The recall index is a doubling buffer (amortized O(1) insert), capped by `MemoryConfig.max_index_rows` (oldest evicted from RAM only; SQLite keeps everything), and recall uses `argpartition` instead of a full sort.*
11. Add structured supervision around always-on loops.
    *`Orchestrator._supervise()` restarts crashed loops with exponential backoff (`SupervisionConfig`), gives up after repeated consecutive failures, and releases a reply waiting on a dead TTS loop.*

## Components That Are Already Strong

These should be preserved while making changes:

- Playback generation-counter stale-audio protection.
- Mic and speaker device supervisors with backoff.
- Windows Job Object child-process cleanup.
- Cross-thread SQLite locking.
- Background memory consolidation concept and tolerant parse-failure behavior.
- Avatar control/audio queue separation.
- Avatar playback-clock handling for `speech_end`.
- Early VAD `SpeechStart` and pre-roll.
- Sentence chunking with action-span protection.

## Final Assessment

The implementation is substantially beyond a prototype: it has real streaming, interruption, recovery, memory, skills, avatar transport, and process-lifecycle handling. The primary architectural risk is not that barge-in merely stops audio; it does cancel the LLM and discard stale downstream output. The remaining cancellation limitation is detached STT/TTS worker computation, which can continue consuming resources after the user interrupts.

The most urgent concrete defect is the mid-stream retry duplication bug. The most urgent engineering gap is the absence of an automated test suite protecting the many ordering-sensitive fixes already present.
