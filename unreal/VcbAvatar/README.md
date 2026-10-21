# VcbAvatar — Unreal MetaHuman ↔ VCB voice bot

Drop-in Unreal 5 plugin that connects your MetaHuman to the VCB voice bot over a
local TCP socket. It plays the streamed speech audio and broadcasts conversation
events (state, emotion, speech start/end, barge-in) plus a smoothed `0..1`
mouth-open value you can bind to a jaw bone or `jawOpen` blendshape as a Phase-0
lip-sync proxy — before you wire up NVIDIA Audio2Face-3D later.

This is the **client** side. The **server** is the Python bridge already built in
VCB (`vcb/avatar/bridge.py`); start VCB with `python main.py --avatar`.

## 1. Install the plugin

1. Copy the `VcbAvatar/` folder (the one containing `VcbAvatar.uplugin`) into your
   Unreal project's `Plugins/` folder, e.g. `MyProject/Plugins/VcbAvatar/`.
2. It's a C++ plugin, so your project must be a **C++ project**. If yours is
   Blueprint-only: in the editor add any C++ class once (Tools → New C++ Class →
   None), which generates the `Source/` + solution files.
3. Right-click your `.uproject` → **Generate Visual Studio project files**.
4. Open the solution and **Build** (Development Editor), or just reopen the
   `.uproject` and let it compile the plugin when prompted.
5. Confirm it's on under **Edit → Plugins → VCB → VCB Avatar** (enabled by default).

## 2. Add it to your MetaHuman actor

1. Open your MetaHuman Blueprint (or the level actor holding it).
2. **Add Component → VCB Avatar** (`UVcbAvatarComponent`).
3. In its Details: set `Host`/`Port` if you changed them (defaults
   `127.0.0.1:8765`), leave `Auto Connect` on.

## 3. Phase-0 test — prove the pipe + audio + a moving jaw

1. Start the backend: `python main.py --avatar` (it listens on `127.0.0.1:8765`).
   By default VCB stops playing audio on the PC so Unreal is the only audio
   source. For a quick A/B you can add `--avatar-local-audio` to hear both.
2. Press **Play** in Unreal. On the component you should see `OnConnected` fire
   (add a Print String) and hear the bot's replies coming from Unreal.
3. Drive the jaw from `OnMouthOpen` (fires every tick, `0..1`):
   - Simplest: on the MetaHuman `Face` skeletal mesh, bind the value to the
     `CTRL_expressions_jawOpen` / `jawOpen` morph or a jaw-bone rotation in the
     **Face Anim Blueprint** (read it via `GetMouthOpen` or cache the delegate
     value into a variable the AnimBP reads).
   - Quick-and-dirty: `OnMouthOpen` → `Set Morph Target` (`jawOpen`) on the face
     mesh. You'll see the mouth open/close in time with speech loudness.

That proves the transport, audio playback, and a live mouth without any lip-sync
tech yet.

## 4. Wire the conversation states / emotions (body + expression)

**Drive the body AnimBP by POLLING, not by binding delegates.** Delegate binds
inside an Animation Blueprint are unreliable (they silently fail to fire — this is
the same gotcha that broke the jaw). In the body AnimBP's **Event Blueprint Update
Animation**, get the component and read these `BlueprintPure` getters into
variables the state machine transitions off:

| Getter | Returns | Suggested use |
|---|---|---|
| `GetAvatarState()` | `EVcbAvatarState` (`Idle`/`Listening`/`Thinking`/`Speaking`) | State-machine selector: attentive lean on `Listening`, a thinking beat on `Thinking`, talk loop on `Speaking` |
| `IsSpeaking()` | `bool` | Blend a talking body loop / gesture layer on top |
| `GetStateSeconds()` | `float` | Time in the current state — fire an idle fidget after N seconds, pace thinking gestures |
| `GetLastEmotions()` | `TArray<FString>` | Pick an expression pose / gesture montage (e.g. `laugh`, `sigh`, `angry`) |
| `GetPrimaryEmotion()` | `FString` | First emotion tag (empty if none) — simplest expression selector |
| `GetSpeakingEnergy()` | `float 0..1` | Steady loudness envelope — scale gesture/hand amplitude while talking (doesn't flap per syllable like `GetMouthOpen`) |
| `GetMouthOpen()` | `float 0..1` | Jaw/`jawOpen` lip-sync proxy (Phase 0) |

Recommended AnimBP wiring (mirrors the working jaw setup):
`Event Blueprint Update Animation` → `Get Owning Component` → `Get Owner` →
`Get Component by Class (Vcb Avatar Component)` → `Get Avatar State` /
`Is Speaking` / `Get State Seconds` → **Set** the matching AnimBP variables. Use
those variables as the state-machine transition rules.

The equivalent Blueprint-assignable events still exist for **level/actor BP**
logic (not the AnimBP): `OnStateChanged(State)`, `OnSpeechStart`/`OnSpeechEnd`,
`OnSpeechChunk(Text, Emotions)`, `OnInterrupt`, `OnMouthOpen(Amount)` — handy for
triggering one-shot montages, SFX, or UI outside the anim graph.

Emotion tags are reliable when VCB uses a tag-capable TTS engine (Orpheus or
Chatterbox-Turbo); other engines won't emit them.

## 5. Later — upgrade lip sync to Audio2Face-3D

Keep everything above. Swap only the mouth driver: run NVIDIA Audio2Face-3D
(Python side, on the GPU) to turn the same audio into ARKit blendshape
weights, stream those into Unreal via **LiveLink**, and drive the MetaHuman face
from the LiveLink pose instead of `OnMouthOpen`. Audio still plays here; only the
face-solve moves.

## Protocol reference

Framed TCP, one message = 5-byte header + payload:

- Header: `>B` msg type, `>I` payload length (big-endian).
- Type `0` = UTF-8 JSON control. Type `1` = `float32` little-endian mono audio.
- On connect the server sends `{"type":"audio_format","sample_rate":..,"channels":1,"dtype":"float32"}`.
- Control JSON types: `state{state}`, `speech_start`, `speech_chunk{text,emotions[]}`,
  `speech_end`, `interrupt`.

## Files

- `VcbAvatar.uplugin` — plugin descriptor (runtime module, enabled by default).
- `Source/VcbAvatar/VcbAvatar.Build.cs` — deps: Sockets, Networking, Json.
- `Public/VcbAvatarClient.h` / `Private/VcbAvatarClient.cpp` — background socket
  thread + frame parser; converts `float32` audio to `int16` PCM and computes RMS.
- `Public/VcbAvatarComponent.h` / `Private/VcbAvatarComponent.cpp` — the
  Blueprint-facing `UVcbAvatarComponent`: audio playback + events + mouth signal.
- `Private/VcbAvatarModule.cpp` — module implementation.
