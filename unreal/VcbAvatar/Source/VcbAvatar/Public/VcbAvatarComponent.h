#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Containers/Queue.h"
#include "VcbAvatarClient.h"
#include "VcbAvatarComponent.generated.h"

class USoundWaveProcedural;
class UAudioComponent;
class UACEAudioCurveSourceComponent;
class IA2FProvider;
class USkeletalMeshComponent;

// Conversation state the body pose/gesture machine keys off. Poll GetAvatarState()
// in the AnimBP's BlueprintUpdateAnimation rather than binding OnStateChanged
// (delegate binds inside an anim instance are flaky - same lesson as the jaw).
UENUM(BlueprintType)
enum class EVcbAvatarState : uint8
{
	Idle       UMETA(DisplayName = "Idle"),
	Listening  UMETA(DisplayName = "Listening"),
	Thinking   UMETA(DisplayName = "Thinking"),
	Speaking   UMETA(DisplayName = "Speaking")
};

// Fired on the game thread; safe to drive Animation Blueprints / Control Rig from these.
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVcbOnState, const FString&, State);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FVcbOnSignal);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FVcbOnSpeechChunk, const FString&, Text, const TArray<FString>&, Emotions);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVcbOnMouth, float, OpenAmount);
// A named body action the LLM/user requested (e.g. "highfive", "wave", "backflip").
// Bind in the body AnimBP's EventGraph and map the name to an AnimMontage to play.
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FVcbOnAction, const FString&, Action);

/**
 * Procedural "aliveness" the body AnimBP reads each frame to drive a sequence-free
 * MetaHuman: breathing, idle sway, head motion and speech-synced accents, all in
 * degrees and 0..1 signals so you can map them straight onto additive Modify Bone
 * nodes. Amplitudes already fold in emotion + speaking energy; tune the overall
 * feel with the "VCB Avatar|Body" properties rather than editing the graph.
 */
USTRUCT(BlueprintType)
struct FVcbBodyDrive
{
	GENERATED_BODY()

	/** Breathing envelope, 0 (exhaled) .. 1 (inhaled). Map to a small chest/spine lift or scale. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float Breath01 = 0.f;

	/** Head pitch in degrees, + = look down. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float HeadNod = 0.f;

	/** Head yaw in degrees, + = turn left. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float HeadTurn = 0.f;

	/** Head roll in degrees, + = tilt right. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float HeadTilt = 0.f;

	/** Torso pitch in degrees, + = lean forward. Spread across the spine chain. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float SpineBend = 0.f;

	/** Torso roll in degrees, + = lean/weight-shift right. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float SpineSway = 0.f;

	/** Torso yaw in degrees, + = twist left. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float SpineTwist = 0.f;

	/** 0..1 impulse that spikes on each spoken phrase and decays; good for gesture accents. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float GesturePulse = 0.f;

	/** 0..1 overall liveliness (idle baseline + speaking energy + arousal); master amplitude. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float MotionEnergy = 0.f;

	/** Emotional valence, -1 (negative/slumped) .. +1 (positive/open). */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float Valence = 0.f;

	/** Emotional arousal, 0 (calm) .. 1 (excited); scales motion speed/amplitude. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Body")
	float Arousal = 0.f;
};

/**
 * Procedural face "aliveness" the face AnimBP reads each frame to layer natural
 * eye motion on top of Audio2Face: blinking, micro-saccades, a slow gaze that can
 * settle on the user/camera, and subtle emotion-driven brows. Audio2Face still
 * owns the mouth/jaw visemes; these fields drive only the eyes, lids and brows so
 * the face never looks frozen or dead-eyed between/over words. Blink and squint are
 * 0..1 ARKit curve amounts; eye gaze is in degrees so you can map it onto the eye
 * bones (or the eyeLook curves) with the same Modify Bone pattern as the body.
 */
USTRUCT(BlueprintType)
struct FVcbFaceDrive
{
	GENERATED_BODY()

	/** Left eyelid close amount, 0 (open) .. 1 (shut). Map to the eyeBlinkLeft curve. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyeBlinkLeft = 0.f;

	/** Right eyelid close amount, 0 (open) .. 1 (shut). Map to the eyeBlinkRight curve. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyeBlinkRight = 0.f;

	/** Eye yaw in degrees, + = look toward the subject's left. Average of both eyes;
	 *  use EyeYawLeft/EyeYawRight instead if you want true convergence (vergence). */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyeYaw = 0.f;

	/** Eye pitch in degrees, + = look up. Rotate the eye bones. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyePitch = 0.f;

	/** Left-eye yaw in degrees (converges on the target). Drive the LEFT eyeLook
	 *  curves from this so both eyes fixate the same point, not stare parallel. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyeYawLeft = 0.f;

	/** Right-eye yaw in degrees (converges on the target). Drive the RIGHT eyeLook curves. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float EyeYawRight = 0.f;

	/** Head/neck yaw turn in degrees, + = subject's left. Apply as a COMPONENT-space
	 *  additive rotation on the neck/head bone so she turns to track wide angles. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float HeadYaw = 0.f;

	/** Head/neck pitch turn in degrees, + = up. Component-space additive on neck/head. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float HeadPitch = 0.f;

	/** Brow raise, 0..1. Map to browInnerUp / browOuterUp (surprise, attentiveness). */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float BrowRaise = 0.f;

	/** Brow furrow, 0..1. Map to browDownLeft / browDownRight (concentration, doubt). */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float BrowFurrow = 0.f;

	/** Eye squint, 0..1. Map to eyeSquintLeft / eyeSquintRight for a subtle live tension. */
	UPROPERTY(BlueprintReadOnly, Category = "VCB Avatar|Face")
	float Squint = 0.f;
};

/**
 * Drop this on the actor that owns your MetaHuman. It connects to the VCB voice
 * bot's local avatar bridge, plays the streamed speech audio, and broadcasts
 * conversation events (state, emotion, speech start/end, barge-in) plus a smooth
 * 0..1 "mouth open" amount you can bind straight to a jaw bone or a jawOpen
 * blendshape for a Phase-0 lip-sync proxy before wiring up Audio2Face.
 */
UCLASS(ClassGroup = (VCB), meta = (BlueprintSpawnableComponent))
class VCBAVATAR_API UVcbAvatarComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UVcbAvatarComponent();

	/** Host of the Python VCB bridge (default localhost). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar")
	FString Host = TEXT("127.0.0.1");

	/** Port of the Python VCB bridge (matches AvatarConfig.port). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar")
	int32 Port = 8765;

	/** Connect automatically on BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar")
	bool bAutoConnect = true;

	/** Attack/release smoothing for the mouth-open signal (higher = snappier). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar", meta = (ClampMin = "1.0"))
	float MouthSmoothingSpeed = 18.f;

	/** Smoothing for the speaking-energy gesture signal (lower = more inertia). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar", meta = (ClampMin = "0.1"))
	float SpeakingEnergySpeed = 3.f;

	/** Scales the RMS-derived mouth-open amount before it is clamped to 0..1. Only
	 *  drives the jaw when Audio2Face is unavailable (the RMS fallback path). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar", meta = (ClampMin = "0.0"))
	float MouthGain = 6.f;

	// -- Procedural body motion (drives a sequence-free ABP_VcbBody) ---------

	/** Master multiplier for all procedural body motion. 0 = perfectly still, 1 = default. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "0.0"))
	float BodyMotionScale = 1.f;

	/** Idle breaths per minute for the breathing envelope. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "1.0"))
	float BreathsPerMinute = 14.f;

	/** Peak idle torso sway/weight-shift, in degrees. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "0.0"))
	float IdleSwayDegrees = 2.5f;

	/** Peak head motion, in degrees (idle look-around + speech nods). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "0.0"))
	float HeadMotionDegrees = 7.f;

	/** Overall size of the talk emphasis (forward lean + nod). Lower = calmer talking. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "0.0"))
	float SpeechAccentScale = 0.6f;

	/** How fast the per-phrase emphasis eases in. Lower = softer/slower (no snap). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Body", meta = (ClampMin = "0.5"))
	float GestureAttackSpeed = 3.f;

	// -- Procedural face motion (blink / eye darts / gaze, layered over A2F) --

	/** Master switch for the procedural face. Off = eyes/blink/brows held neutral
	 *  (Audio2Face still drives the mouth). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face")
	bool bFaceMotion = true;

	/** Overall multiplier for eye-gaze and brow amplitude (blink timing is separate). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "0.0"))
	float FaceMotionScale = 1.f;

	/** Average blinks per minute (real humans ~15-20). Intervals are randomised
	 *  around this, with the odd double-blink. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "0.0"))
	float BlinksPerMinute = 18.f;

	/** Amplitude of the quick idle eye darts (saccades), in degrees. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "0.0"))
	float EyeDartDegrees = 4.f;

	/** Maximum eye deflection from centre, in degrees; clamps gaze + look-at. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "0.0"))
	float EyeGazeRangeDegrees = 16.f;

	/** Bias the gaze toward the player camera so the avatar "looks at the user".
	 *  Saccades still play on top; while thinking she naturally glances away. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face")
	bool bGazeAtUser = true;

	/** Optional explicit thing to look at (e.g. a placed CameraActor). If set, gaze
	 *  aims here instead of the auto player camera. Leave empty to track the player view. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face")
	TObjectPtr<AActor> GazeTargetActor = nullptr;

	/** Flip the horizontal gaze if she looks left when she should look right. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face")
	bool bInvertGazeYaw = false;

	/** Flip the vertical gaze if she looks up when she should look down. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face")
	bool bInvertGazePitch = false;

	/** Horizontal aim correction, in degrees, for when the mesh faces a different
	 *  way than the actor. Nudge until she centres on the target. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "-90", ClampMax = "90"))
	float GazeYawTrimDeg = 0.f;

	/** Vertical aim correction, in degrees; raise/lower until her eyes meet the target. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "-90", ClampMax = "90"))
	float GazePitchTrimDeg = 0.f;

	/** Approx height of the eyes above the actor origin, in cm; used to aim the
	 *  look-at at the camera. Tune to your MetaHuman's eye height. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face", meta = (ClampMin = "0.0"))
	float EyeHeightCm = 160.f;

	/** Point the gaze at a specific actor at runtime (empty = back to the player view). */
	UFUNCTION(BlueprintCallable, Category = "VCB Avatar|Face")
	void SetGazeTargetActor(AActor* Target) { GazeTargetActor = Target; }

	/** Turn the HEAD/neck toward the target for wide angles the eyes can't reach,
	 *  like a real person. Enable this ONLY after you wire the head Modify Bone in
	 *  the AnimBP, otherwise the eyes give up their share to a head that never moves. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head")
	bool bHeadTracking = false;

	/** Share of the aim the head takes (rest is covered by the eyes). 0 = eyes only,
	 *  1 = head does everything. ~0.6-0.75 reads natural. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float HeadFollowFraction = 0.7f;

	/** Max head/neck yaw turn in degrees (human neck ~70). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head", meta = (ClampMin = "0.0", ClampMax = "90.0"))
	float HeadYawLimitDegrees = 65.f;

	/** Max head/neck pitch turn in degrees. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head", meta = (ClampMin = "0.0", ClampMax = "80.0"))
	float HeadPitchLimitDegrees = 30.f;

	/** How quickly the head chases the target (interp speed). Lower = slower, heavier. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head", meta = (ClampMin = "0.5", ClampMax = "30.0"))
	float HeadTurnSpeed = 7.f;

	/** Converge both eyes on the actual target point (toe-in) instead of staring
	 *  parallel. Needs the LEFT/RIGHT eyeLook curves driven from EyeYawLeft/Right. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head")
	bool bEyeVergence = true;

	/** Head bone used for the neck turn + gaze origin (MetaHuman = "head"). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head")
	FName HeadBoneName = TEXT("head");

	/** Left eye bone, for per-eye vergence (MetaHuman = "FACIAL_L_Eye"). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head")
	FName LeftEyeBoneName = TEXT("FACIAL_L_Eye");

	/** Right eye bone, for per-eye vergence (MetaHuman = "FACIAL_R_Eye"). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Face|Head")
	FName RightEyeBoneName = TEXT("FACIAL_R_Eye");

	/** Stream the speech audio into NVIDIA Audio2Face-3D to drive real MetaHuman
	 *  visemes/blendshapes. If the provider isn't available (e.g. no NVIDIA GPU),
	 *  the component transparently falls back to the RMS loudness jaw. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Audio2Face")
	bool bUseAudio2Face = true;

	/** Name of the Audio2Face-3D provider to use. "LocalA2F-Claire" is the local,
	 *  on-GPU model; if it isn't registered the component auto-picks another local one. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Audio2Face")
	FName A2FProviderName = FName(TEXT("LocalA2F-Claire"));

	/** Silence fed through Audio2Face at the start of each utterance so the model's
	 *  warm-up (weak first frames) happens before the first word instead of on it.
	 *  Adds this much latency to the start of speech. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Audio2Face", meta = (ClampMin = "0", ClampMax = "1000"))
	int32 A2FPrimeSilenceMs = 200;

	/** Delays the audible voice this many ms relative to the Audio2Face animation.
	 *  Raise it if the mouth still lags the audio; 0 disables. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCB Avatar|Audio2Face", meta = (ClampMin = "0", ClampMax = "1000"))
	int32 AudioSyncOffsetMs = 0;

	/** True while speech audio is being driven through Audio2Face-3D (vs the RMS fallback). */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Audio2Face")
	bool IsAudio2FaceActive() const { return bA2FActive; }

	UFUNCTION(BlueprintCallable, Category = "VCB Avatar")
	void Connect();

	UFUNCTION(BlueprintCallable, Category = "VCB Avatar")
	void Disconnect();

	/** Latest smoothed mouth-open amount (0..1); also available via OnMouthOpen. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	float GetMouthOpen() const { return MouthOpen; }

	/** Current conversation state. Poll this in the AnimBP to drive body poses. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	EVcbAvatarState GetAvatarState() const { return AvatarState; }

	/** Convenience: true while the avatar is speaking. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	bool IsSpeaking() const { return AvatarState == EVcbAvatarState::Speaking; }

	/** Seconds elapsed in the current state; handy for timing idle/thinking gestures. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	float GetStateSeconds() const { return StateSeconds; }

	/** Emotion tags from the most recent speech chunk (e.g. laugh, sigh). */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	const TArray<FString>& GetLastEmotions() const { return LastEmotions; }

	/** First tag of the most recent speech chunk, or empty if none. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	FString GetPrimaryEmotion() const { return LastEmotions.Num() > 0 ? LastEmotions[0] : FString(); }

	/** Slowly-smoothed speech loudness (0..1) for scaling gesture amplitude while
	 *  talking; steadier than GetMouthOpen, which flaps per syllable. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar")
	float GetSpeakingEnergy() const { return SpeakingEnergy; }

	/** One node with every procedural body signal for this frame. Read it in the
	 *  body AnimBP and map the fields onto additive Modify Bone nodes. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Body")
	FVcbBodyDrive GetBodyDrive() const { return BodyDrive; }

	/** One node with every procedural face signal for this frame (blink, eye gaze,
	 *  brows). Read it in the face AnimBP and layer it over the Audio2Face mouth. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Face")
	FVcbFaceDrive GetFaceDrive() const { return FaceDrive; }

	/** World point she is currently trying to look at (player's eyes / gaze target).
	 *  Feed this to an AnimGraph "Look At" node if you prefer that over Modify Bone. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Face")
	FVector GetGazeTargetWorldLocation() const { return GazeTargetWorld; }

	/** True when there is a valid gaze target this frame. */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Face")
	bool HasGazeTarget() const { return bHasGazeTarget; }

	/** Name of the most recent requested body action (also delivered via OnAction). */
	UFUNCTION(BlueprintPure, Category = "VCB Avatar|Body")
	FString GetLastAction() const { return LastAction; }

	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSignal OnConnected;

	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSignal OnDisconnected;

	/** State is one of: idle, listening, thinking, speaking. */
	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnState OnStateChanged;

	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSignal OnSpeechStart;

	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSpeechChunk OnSpeechChunk;

	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSignal OnSpeechEnd;

	/** User barged in: stop mouth/gesture immediately. */
	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnSignal OnInterrupt;

	/** Smoothed 0..1 mouth-open amount, broadcast every tick while speaking. */
	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar")
	FVcbOnMouth OnMouthOpen;

	/** A named body action was requested (e.g. "wave", "highfive"); play its montage. */
	UPROPERTY(BlueprintAssignable, Category = "VCB Avatar|Body")
	FVcbOnAction OnAction;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
	                           FActorComponentTickFunction* ThisTickFunction) override;

private:
	void HandleJson(const FString& Json);
	void EnsureAudio(int32 SampleRate);
	void SetAvatarState(const FString& Name);

	// -- Procedural body motion ---------------------------------------------
	/** Advance breathing/sway/head/accent signals and fold in emotion + energy. */
	void UpdateProceduralBody(float DeltaTime);
	/** Re-derive the valence/arousal targets from the latest emotion tags. */
	void UpdateEmotionTargets();

	// -- Procedural face motion ---------------------------------------------
	/** Advance blinking, eye saccades, gaze/look-at and emotion-driven brows. */
	void UpdateProceduralFace(float DeltaTime);
	/** Reset all procedural face state to a neutral, open-eyed rest. */
	void ResetFaceState();
	/** Find (and cache) the owner's skeletal mesh that carries the head/eye bones. */
	USkeletalMeshComponent* ResolveFaceMesh();

	// -- Audio2Face-3D streaming --------------------------------------------
	/** Find the A2F provider and ensure a curve-source component exists. Returns
	 *  true if A2F is usable; false means stay on the RMS fallback. */
	bool InitAudio2Face();
	/** Open a fresh A2F stream for the current speaking turn. */
	void BeginA2FStream(int32 SampleRate);
	/** Queue one frame of int16 mono audio: passthrough for playback + resample for A2F. */
	void PumpA2FAudio(const uint8* Pcm16Bytes, int32 NumBytes, int32 SampleRate);
	/** Send buffered 16 kHz samples to the provider, paced near real time so local
	 *  inference doesn't burst and starve rendering (per NVIDIA guidance). */
	void FlushA2FSamples();
	/** All audio for the turn is in: append the end pad and let the buffer drain
	 *  through the paced flush; the stream closes itself once empty. */
	void RequestA2FStreamEnd();
	/** Close the current A2F stream now, discarding anything still buffered
	 *  (interrupt / disconnect / drained pending end). */
	void EndA2FStream();
	/** Append int16 samples resampled from SrcRate to 16 kHz into ResampleOut. */
	void ResampleTo16k(const int16* In, int32 NumSamples, int32 SrcRate);
	void ResetResampler();

	UPROPERTY(Transient)
	TObjectPtr<USoundWaveProcedural> ProceduralWave = nullptr;

	UPROPERTY(Transient)
	TObjectPtr<UAudioComponent> AudioComp = nullptr;

	TUniquePtr<FVcbAvatarClient> Client;
	TQueue<FVcbAvatarMessage, EQueueMode::Spsc> Incoming;

	// Audio2Face-3D: when active, speech audio is streamed to the A2F provider and
	// the resulting blendshapes drive the MetaHuman face; the RMS jaw path is unused.
	bool bA2FActive = false;       // provider found + curve source ready
	bool bA2FStreamOpen = false;   // a stream is open for the current speaking turn
	bool bA2FEndPending = false;   // end pad queued; close the stream once drained
	bool bSentInitialAudio = false;
	int32 MinInitialSamples = 0;
	// Inference input pacing (near real time after an initial burst).
	double A2FSendStart = 0.0;     // wall clock of the initial send
	int64 A2FSamplesSent = 0;      // 16 kHz samples sent so far this stream
	int64 A2FOrigBytesSent = 0;    // native-rate playback bytes released so far
	// Native-rate playback audio buffered until it is enqueued paired 1:1 with its
	// matching 16 kHz inference chunk (ACE requires enqueue-then-send per chunk so
	// the voice and the blendshapes stay locked together).
	TArray<uint8> PendingOriginal;
	IA2FProvider* A2FProvider = nullptr;
	void* A2FStream = nullptr;     // IA2FProvider::IA2FStream* (opaque to keep the header light)

	UPROPERTY(Transient)
	TObjectPtr<UACEAudioCurveSourceComponent> CurveSource = nullptr;

	// Stateful linear resampler (source rate -> 16 kHz) for the A2F input stream.
	TArray<int16> ResampleOut;
	double ResamplePos = 0.0;
	int16 ResamplePrev = 0;

	int32 CurrentSampleRate = 0;
	float MouthTarget = 0.f;   // loudness of the audio frame currently "playing"
	float MouthOpen = 0.f;     // smoothed value that is broadcast
	float SpeakingEnergy = 0.f; // slow envelope of loudness for gesture amplitude

	// Polled by the AnimBP to drive the body state machine / gestures.
	EVcbAvatarState AvatarState = EVcbAvatarState::Idle;
	float StateSeconds = 0.f;
	TArray<FString> LastEmotions;

	// Procedural body-motion state, published each frame via GetBodyDrive().
	FVcbBodyDrive BodyDrive;
	float BreathPhase = 0.f;    // radians, advances at BreathsPerMinute
	float NoiseTime = 0.f;      // clock for the idle Perlin sway
	float GesturePulse = 0.f;   // smoothed accent applied to the body
	float GesturePulseTarget = 0.f;  // re-armed to 1 per phrase; GesturePulse chases it
	float MotionEnergy = 0.f;   // smoothed master amplitude
	float ValenceTarget = 0.f;  // emotion targets the published values chase
	float ArousalTarget = 0.f;
	FString LastAction;

	// Procedural face-motion state, published each frame via GetFaceDrive().
	FVcbFaceDrive FaceDrive;
	float BlinkClock = 0.f;        // time since the last blink began
	float BlinkInterval = 3.f;     // randomised seconds until the next blink
	float BlinkPhase = -1.f;       // <0 = eyes open; 0..1 = a blink in progress
	int32 PendingBlinks = 0;       // queued extra blinks (double-blink)
	float GazeYaw = 0.f;           // current eye yaw/pitch (deg), chase the target
	float GazePitch = 0.f;
	float GazeYawLeft = 0.f;       // current per-eye yaw (deg) for vergence
	float GazeYawRight = 0.f;
	float HeadYawVal = 0.f;        // smoothed head/neck turn (deg)
	float HeadPitchVal = 0.f;
	FVector GazeTargetWorld = FVector::ZeroVector;  // world point she looks at
	bool bHasGazeTarget = false;
	TWeakObjectPtr<USkeletalMeshComponent> CachedFaceMesh;  // mesh holding the head/eye bones
	float GazeTargetYaw = 0.f;     // saccade target the eyes jump to
	float GazeTargetPitch = 0.f;
	float SaccadeClock = 0.f;      // time since the last saccade
	float SaccadeInterval = 1.f;   // randomised seconds until the next saccade
	float FaceNoiseTime = 0.f;     // clock for slow brow/gaze drift
	float BrowRaiseVal = 0.f;      // smoothed brow signals
	float BrowFurrowVal = 0.f;
	float SquintVal = 0.f;

	// Per-frame loudness queued and consumed at real time, so the mouth tracks
	// the audio as it plays instead of spiking once when a whole chunk arrives.
	TArray<float> PendingRms;
	TArray<float> PendingDur;
	float MouthClock = 0.f;
};
