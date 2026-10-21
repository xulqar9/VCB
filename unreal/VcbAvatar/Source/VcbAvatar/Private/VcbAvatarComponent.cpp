#include "VcbAvatarComponent.h"

#include "Sound/SoundWaveProcedural.h"
#include "Components/AudioComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Camera/PlayerCameraManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "GameFramework/Actor.h"

// NVIDIA ACE / Audio2Face-3D
#include "A2FProvider.h"
#include "ACEAudioCurveSourceComponent.h"

DEFINE_LOG_CATEGORY_STATIC(LogVcbAvatarComp, Log, All);

// Audio2Face-3D requires 16 kHz mono PCM16 input.
static constexpr int32 kA2FSampleRate = 16000;

UVcbAvatarComponent::UVcbAvatarComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UVcbAvatarComponent::BeginPlay()
{
	Super::BeginPlay();
	InitAudio2Face();
	ResetFaceState();
	if (bAutoConnect)
	{
		Connect();
	}
}

void UVcbAvatarComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Disconnect();
	Super::EndPlay(EndPlayReason);
}

void UVcbAvatarComponent::Connect()
{
	if (Client.IsValid())
	{
		return;
	}
	// Clear anything left from a previous session.
	FVcbAvatarMessage Drain;
	while (Incoming.Dequeue(Drain)) {}

	Client = MakeUnique<FVcbAvatarClient>(Host, Port, &Incoming);
	Client->StartThread();
}

void UVcbAvatarComponent::Disconnect()
{
	if (Client.IsValid())
	{
		Client->StopThread();
		Client.Reset();
	}
	EndA2FStream();
	if (AudioComp)
	{
		AudioComp->Stop();
	}
	if (ProceduralWave)
	{
		ProceduralWave->ResetAudio();
	}
	MouthTarget = 0.f;
	MouthOpen = 0.f;
	SpeakingEnergy = 0.f;
	PendingRms.Reset();
	PendingDur.Reset();
	MouthClock = 0.f;
	AvatarState = EVcbAvatarState::Idle;
	StateSeconds = 0.f;
	LastEmotions.Reset();
	GesturePulse = 0.f;
	GesturePulseTarget = 0.f;
	ValenceTarget = 0.f;
	ArousalTarget = 0.f;
	LastAction.Reset();
	ResetFaceState();
}

void UVcbAvatarComponent::EnsureAudio(int32 SampleRate)
{
	if (SampleRate <= 0)
	{
		SampleRate = 24000;
	}
	if (ProceduralWave && CurrentSampleRate == SampleRate && AudioComp)
	{
		return;   // already streaming at this rate; don't flush queued audio
	}

	if (AudioComp)
	{
		AudioComp->Stop();
		AudioComp = nullptr;
	}

	CurrentSampleRate = SampleRate;
	ProceduralWave = NewObject<USoundWaveProcedural>(this);
	ProceduralWave->SetSampleRate(SampleRate);
	ProceduralWave->NumChannels = 1;
	ProceduralWave->Duration = INDEFINITELY_LOOPING_DURATION;
	ProceduralWave->bLooping = false;

	// 2D playback so the voice isn't spatially attenuated. Do not auto-destroy:
	// we keep feeding the same procedural wave for the whole session.
	AudioComp = UGameplayStatics::SpawnSound2D(
		this, ProceduralWave, 1.f, 1.f, 0.f, nullptr, false, /*bAutoDestroy*/ false);
}

void UVcbAvatarComponent::TickComponent(float DeltaTime, ELevelTick TickType,
                                        FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	FVcbAvatarMessage Msg;
	while (Incoming.Dequeue(Msg))
	{
		switch (Msg.Type)
		{
		case EVcbMsgType::Connected:
			OnConnected.Broadcast();
			break;
		case EVcbMsgType::Disconnected:
			EndA2FStream();
			MouthTarget = 0.f;
			PendingRms.Reset();
			PendingDur.Reset();
			MouthClock = 0.f;
			AvatarState = EVcbAvatarState::Idle;
			StateSeconds = 0.f;
			LastEmotions.Reset();
			if (ProceduralWave)
			{
				ProceduralWave->ResetAudio();   // start clean if it reconnects
			}
			OnDisconnected.Broadcast();
			break;
		case EVcbMsgType::Audio:
			if (Msg.Pcm16.Num() > 0)
			{
				const int32 Rate = CurrentSampleRate > 0 ? CurrentSampleRate : 24000;
				if (bA2FActive)
				{
					if (bA2FStreamOpen && bA2FEndPending)
					{
						EndA2FStream();   // previous turn was ending; new audio = new turn
					}
					if (!bA2FStreamOpen)
					{
						BeginA2FStream(Rate);
					}
					PumpA2FAudio(Msg.Pcm16.GetData(), Msg.Pcm16.Num(), Rate);
				}
				else
				{
					EnsureAudio(Rate);
					if (ProceduralWave)
					{
						ProceduralWave->QueueAudio(Msg.Pcm16.GetData(), Msg.Pcm16.Num());
					}
				}
				// Track loudness for the body-gesture signals (and the RMS jaw fallback),
				// consumed in step with playback below.
				PendingRms.Add(Msg.Rms);
				PendingDur.Add((Msg.Pcm16.Num() / 2) / static_cast<float>(Rate));
			}
			break;
		case EVcbMsgType::Json:
			HandleJson(Msg.Json);
			break;
		}
	}

	// Push buffered 16 kHz audio to Audio2Face-3D, paced near real time.
	FlushA2FSamples();

	// Walk a real-time playhead over the queued frames so the jaw follows the
	// audio as it plays: consume one frame's loudness per its own duration.
	if (PendingDur.Num() > 0)
	{
		MouthClock += DeltaTime;
		while (PendingDur.Num() > 0 && MouthClock >= PendingDur[0])
		{
			MouthClock -= PendingDur[0];
			MouthTarget = FMath::Clamp(PendingRms[0] * MouthGain, 0.f, 1.f);
			PendingRms.RemoveAt(0);
			PendingDur.RemoveAt(0);
		}
	}
	else
	{
		// Nothing queued: close the mouth.
		MouthClock = 0.f;
		MouthTarget = FMath::FInterpTo(MouthTarget, 0.f, DeltaTime, 8.f);
	}

	// Smooth the mouth toward the current loudness and broadcast it.
	const float Alpha = FMath::Clamp(DeltaTime * MouthSmoothingSpeed, 0.f, 1.f);
	MouthOpen = FMath::Lerp(MouthOpen, MouthTarget, Alpha);
	OnMouthOpen.Broadcast(MouthOpen);

	// Slow envelope of loudness for scaling body/hand gesture amplitude.
	SpeakingEnergy = FMath::FInterpTo(SpeakingEnergy, MouthTarget, DeltaTime, SpeakingEnergySpeed);

	// Advance the procedural body signals (breathing, sway, head, accents).
	UpdateProceduralBody(DeltaTime);

	// Advance the procedural face signals (blink, eye saccades, gaze, brows).
	UpdateProceduralFace(DeltaTime);

	// Age the current state so the AnimBP can time idle/thinking gestures.
	StateSeconds += DeltaTime;
}

void UVcbAvatarComponent::HandleJson(const FString& Json)
{
	TSharedPtr<FJsonObject> Obj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Json);
	if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
	{
		UE_LOG(LogVcbAvatarComp, Warning, TEXT("bad JSON from bridge: %s"), *Json);
		return;
	}

	const FString Type = Obj->GetStringField(TEXT("type"));
	if (Type == TEXT("audio_format"))
	{
		const int32 Rate = Obj->HasField(TEXT("sample_rate"))
			? static_cast<int32>(Obj->GetNumberField(TEXT("sample_rate")))
			: 24000;
		CurrentSampleRate = Rate;
		if (!bA2FActive)
		{
			EnsureAudio(Rate);   // fallback path plays the audio itself
		}
	}
	else if (Type == TEXT("state"))
	{
		const FString StateName = Obj->GetStringField(TEXT("state"));
		SetAvatarState(StateName);
		OnStateChanged.Broadcast(StateName);
	}
	else if (Type == TEXT("speech_start"))
	{
		GesturePulseTarget = 1.f;   // arm a gesture accent as speech begins
		OnSpeechStart.Broadcast();
	}
	else if (Type == TEXT("speech_chunk"))
	{
		const FString Text = Obj->GetStringField(TEXT("text"));
		TArray<FString> Emotions;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (Obj->TryGetArrayField(TEXT("emotions"), Arr) && Arr)
		{
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				Emotions.Add(V->AsString());
			}
		}
		LastEmotions = Emotions;
		UpdateEmotionTargets();   // re-derive valence/arousal from the new tags
		GesturePulseTarget = 1.f;       // accent each fresh phrase
		OnSpeechChunk.Broadcast(Text, Emotions);
	}
	else if (Type == TEXT("speech_flush"))
	{
		// All audio for this turn has been shipped: queue the end pad and let the
		// paced flush drain it. The stream then closes itself while playback still
		// has headroom, so the model's held tail can't glitch the last word, and
		// the blocking end-of-stream call never lands mid-speech on a full buffer.
		RequestA2FStreamEnd();
	}
	else if (Type == TEXT("speech_end"))
	{
		RequestA2FStreamEnd();   // no-op if speech_flush already queued the end
		OnSpeechEnd.Broadcast();
	}
	else if (Type == TEXT("interrupt"))
	{
		EndA2FStream();
		if (CurveSource)
		{
			CurveSource->Stop();            // stop A2F audio + animation instantly
		}
		if (ProceduralWave)
		{
			ProceduralWave->ResetAudio();   // drop queued speech instantly (fallback)
		}
		MouthTarget = 0.f;
		PendingRms.Reset();
		PendingDur.Reset();
		MouthClock = 0.f;
		OnInterrupt.Broadcast();
	}
	else if (Type == TEXT("action"))
	{
		FString Name;
		if (Obj->TryGetStringField(TEXT("name"), Name) && !Name.IsEmpty())
		{
			LastAction = Name;
			UE_LOG(LogVcbAvatarComp, Log, TEXT("body action requested: %s"), *Name);
			OnAction.Broadcast(Name);
		}
	}
}

void UVcbAvatarComponent::SetAvatarState(const FString& Name)
{
	EVcbAvatarState NewState = AvatarState;
	if (Name == TEXT("idle"))            NewState = EVcbAvatarState::Idle;
	else if (Name == TEXT("listening")) NewState = EVcbAvatarState::Listening;
	else if (Name == TEXT("thinking"))  NewState = EVcbAvatarState::Thinking;
	else if (Name == TEXT("speaking"))  NewState = EVcbAvatarState::Speaking;
	else
	{
		UE_LOG(LogVcbAvatarComp, Warning, TEXT("unknown avatar state: %s"), *Name);
		return;
	}
	if (NewState != AvatarState)
	{
		AvatarState = NewState;
		StateSeconds = 0.f;
	}
}

// ---------------------------------------------------------------------------
// Procedural body motion
// ---------------------------------------------------------------------------

void UVcbAvatarComponent::UpdateEmotionTargets()
{
	// Map the loudest/first emotion tag to a coarse valence/arousal the body leans
	// into. Unknown or absent tags relax toward neutral. Kept intentionally small
	// so posture nudges rather than caricatures.
	struct FVA { float V; float A; };
	static const TMap<FString, FVA> kTable = {
		{TEXT("laugh"),   {  0.9f, 0.85f}},
		{TEXT("giggle"),  {  0.8f, 0.7f }},
		{TEXT("chuckle"), {  0.6f, 0.5f }},
		{TEXT("gasp"),    {  0.1f, 0.9f }},
		{TEXT("sigh"),    { -0.4f, 0.2f }},
		{TEXT("groan"),   { -0.6f, 0.4f }},
		{TEXT("yawn"),    { -0.2f, 0.1f }},
		{TEXT("sniffle"), { -0.7f, 0.3f }},
		{TEXT("cough"),   {  0.0f, 0.4f }},
		{TEXT("gulp"),    { -0.3f, 0.6f }},
	};

	ValenceTarget = 0.f;
	ArousalTarget = 0.f;
	for (const FString& Tag : LastEmotions)
	{
		if (const FVA* Found = kTable.Find(Tag.ToLower()))
		{
			// Take the strongest reading across the tags on this phrase.
			if (FMath::Abs(Found->V) > FMath::Abs(ValenceTarget)) ValenceTarget = Found->V;
			ArousalTarget = FMath::Max(ArousalTarget, Found->A);
		}
	}
}

void UVcbAvatarComponent::UpdateProceduralBody(float DeltaTime)
{
	const float Scale = FMath::Max(0.f, BodyMotionScale);
	NoiseTime += DeltaTime;

	// Breathing: smooth 0..1 cosine at the configured rate.
	const float BreathHz = FMath::Max(0.1f, BreathsPerMinute) / 60.f;
	BreathPhase += DeltaTime * BreathHz * 2.f * PI;
	BreathPhase = FMath::Fmod(BreathPhase, 2.f * PI);
	const float Breath01 = 0.5f * (1.f - FMath::Cos(BreathPhase));

	// Gesture accent: target is re-armed to 1 per phrase and decays; the applied
	// value chases it with a soft attack so emphasis eases in instead of snapping.
	GesturePulseTarget = FMath::FInterpTo(GesturePulseTarget, 0.f, DeltaTime, 2.0f);
	GesturePulse = FMath::FInterpTo(GesturePulse, GesturePulseTarget, DeltaTime,
		FMath::Max(0.5f, GestureAttackSpeed));

	// Emotion values chase their targets; relax when speech (and its tags) stop.
	const bool bSpeaking = AvatarState == EVcbAvatarState::Speaking;
	if (!bSpeaking)
	{
		ValenceTarget = FMath::FInterpTo(ValenceTarget, 0.f, DeltaTime, 0.5f);
		ArousalTarget = FMath::FInterpTo(ArousalTarget, 0.f, DeltaTime, 0.5f);
	}
	BodyDrive.Valence = FMath::FInterpTo(BodyDrive.Valence, ValenceTarget, DeltaTime, 2.f);
	BodyDrive.Arousal = FMath::FInterpTo(BodyDrive.Arousal, ArousalTarget, DeltaTime, 2.f);

	// Master liveliness: idle baseline + speaking loudness + arousal.
	const float EnergyTarget = FMath::Clamp(
		0.15f + 0.75f * SpeakingEnergy + 0.30f * BodyDrive.Arousal, 0.f, 1.f);
	MotionEnergy = FMath::FInterpTo(MotionEnergy, EnergyTarget, DeltaTime, 3.f);

	// Slow, decorrelated idle noise for the torso/head so nothing looks looped.
	const float N1 = FMath::PerlinNoise1D(NoiseTime * 0.13f + 11.f);
	const float N2 = FMath::PerlinNoise1D(NoiseTime * 0.11f + 53.f);
	const float N3 = FMath::PerlinNoise1D(NoiseTime * 0.17f + 97.f);

	const float SwayAmp = IdleSwayDegrees * Scale;
	const float HeadAmp = HeadMotionDegrees * Scale;
	const float LiveAmp = 0.5f + 0.5f * MotionEnergy;
	const float Accent = FMath::Max(0.f, SpeechAccentScale) * Scale;

	// Torso: idle sway/twist + a small forward lean while emphatic, plus a slump
	// when valence is negative and a subtle breathing lift.
	BodyDrive.SpineSway  = N1 * SwayAmp * LiveAmp;
	BodyDrive.SpineTwist = N2 * SwayAmp * 0.8f * LiveAmp;
	BodyDrive.SpineBend  = N3 * SwayAmp * 0.4f
		+ (bSpeaking ? (SpeakingEnergy * 2.5f + GesturePulse * 2.0f) * Accent : 0.f)
		+ FMath::Max(0.f, -BodyDrive.Valence) * 3.5f * Scale
		- Breath01 * 0.6f * Scale;

	// Head: idle look-around + speech-synced nods, lifted by positive valence and
	// dropped by negative. Signs assume UE bone axes; flip in the AnimBP if needed.
	BodyDrive.HeadTurn = N1 * HeadAmp * LiveAmp;
	BodyDrive.HeadTilt = N2 * HeadAmp * 0.45f * LiveAmp + BodyDrive.Valence * 1.5f * Scale;
	BodyDrive.HeadNod  = N3 * HeadAmp * 0.5f
		+ (bSpeaking ? (SpeakingEnergy * 4.0f + GesturePulse * 3.0f) * Accent : 0.f)
		- BodyDrive.Valence * 3.0f * Scale;

	BodyDrive.Breath01     = Breath01;
	BodyDrive.GesturePulse = GesturePulse;
	BodyDrive.MotionEnergy = MotionEnergy;
}

// ---------------------------------------------------------------------------
// Procedural face motion (blink / eye saccades / gaze / brows)
// ---------------------------------------------------------------------------

void UVcbAvatarComponent::ResetFaceState()
{
	FaceDrive = FVcbFaceDrive();
	BlinkClock = 0.f;
	BlinkPhase = -1.f;
	PendingBlinks = 0;
	BlinkInterval = FMath::FRandRange(2.f, 5.f);
	GazeYaw = GazePitch = 0.f;
	GazeYawLeft = GazeYawRight = 0.f;
	HeadYawVal = HeadPitchVal = 0.f;
	bHasGazeTarget = false;
	GazeTargetYaw = GazeTargetPitch = 0.f;
	SaccadeClock = 0.f;
	SaccadeInterval = FMath::FRandRange(0.7f, 2.2f);
	BrowRaiseVal = BrowFurrowVal = SquintVal = 0.f;
}

USkeletalMeshComponent* UVcbAvatarComponent::ResolveFaceMesh()
{
	if (USkeletalMeshComponent* Cached = CachedFaceMesh.Get())
	{
		return Cached;
	}
	const AActor* Owner = GetOwner();
	if (!Owner)
	{
		return nullptr;
	}
	TArray<USkeletalMeshComponent*> Meshes;
	Owner->GetComponents<USkeletalMeshComponent>(Meshes);
	// Prefer the mesh that actually carries the head bone (the MetaHuman body/face).
	for (USkeletalMeshComponent* M : Meshes)
	{
		if (M && M->DoesSocketExist(HeadBoneName))
		{
			CachedFaceMesh = M;
			return M;
		}
	}
	if (Meshes.Num() > 0)
	{
		CachedFaceMesh = Meshes[0];
		return Meshes[0];
	}
	return nullptr;
}

void UVcbAvatarComponent::UpdateProceduralFace(float DeltaTime)
{
	if (!bFaceMotion)
	{
		// Relax everything to a neutral, open-eyed rest.
		FaceDrive.EyeBlinkLeft  = FMath::FInterpTo(FaceDrive.EyeBlinkLeft, 0.f, DeltaTime, 12.f);
		FaceDrive.EyeBlinkRight = FMath::FInterpTo(FaceDrive.EyeBlinkRight, 0.f, DeltaTime, 12.f);
		FaceDrive.EyeYaw    = FMath::FInterpTo(FaceDrive.EyeYaw, 0.f, DeltaTime, 6.f);
		FaceDrive.EyePitch  = FMath::FInterpTo(FaceDrive.EyePitch, 0.f, DeltaTime, 6.f);
		FaceDrive.EyeYawLeft  = FMath::FInterpTo(FaceDrive.EyeYawLeft, 0.f, DeltaTime, 6.f);
		FaceDrive.EyeYawRight = FMath::FInterpTo(FaceDrive.EyeYawRight, 0.f, DeltaTime, 6.f);
		FaceDrive.HeadYaw   = FMath::FInterpTo(FaceDrive.HeadYaw, 0.f, DeltaTime, 5.f);
		FaceDrive.HeadPitch = FMath::FInterpTo(FaceDrive.HeadPitch, 0.f, DeltaTime, 5.f);
		HeadYawVal = FaceDrive.HeadYaw; HeadPitchVal = FaceDrive.HeadPitch;
		FaceDrive.BrowRaise = FMath::FInterpTo(FaceDrive.BrowRaise, 0.f, DeltaTime, 6.f);
		FaceDrive.BrowFurrow= FMath::FInterpTo(FaceDrive.BrowFurrow, 0.f, DeltaTime, 6.f);
		FaceDrive.Squint    = FMath::FInterpTo(FaceDrive.Squint, 0.f, DeltaTime, 6.f);
		return;
	}

	const float Scale = FMath::Max(0.f, FaceMotionScale);
	FaceNoiseTime += DeltaTime;

	// -- Blinking ----------------------------------------------------------
	// Random intervals around BlinksPerMinute, with the occasional double blink.
	// A blink is a fast close and a slightly slower open (~0.14s total).
	if (BlinkPhase < 0.f)
	{
		BlinkClock += DeltaTime;
		if (BlinkClock >= BlinkInterval)
		{
			BlinkPhase = 0.f;   // start a blink
			BlinkClock = 0.f;
		}
	}
	if (BlinkPhase >= 0.f)
	{
		constexpr float kBlinkDuration = 0.14f;
		BlinkPhase += DeltaTime / kBlinkDuration;
		// Asymmetric shape: shut by ~0.4 of the way in, then open.
		float Lid;
		if (BlinkPhase < 0.4f)
		{
			Lid = BlinkPhase / 0.4f;
		}
		else
		{
			Lid = FMath::Max(0.f, 1.f - (BlinkPhase - 0.4f) / 0.6f);
		}
		FaceDrive.EyeBlinkLeft  = FMath::Clamp(Lid, 0.f, 1.f);
		FaceDrive.EyeBlinkRight = FaceDrive.EyeBlinkLeft;
		if (BlinkPhase >= 1.f)
		{
			FaceDrive.EyeBlinkLeft = FaceDrive.EyeBlinkRight = 0.f;
			BlinkPhase = -1.f;
			if (PendingBlinks > 0)
			{
				--PendingBlinks;
				BlinkInterval = 0.12f;   // quick second blink
			}
			else
			{
				const float Base = BlinksPerMinute > 0.f ? 60.f / BlinksPerMinute : 4.f;
				BlinkInterval = Base * FMath::FRandRange(0.6f, 1.5f);
				if (FMath::FRand() < 0.12f)
				{
					PendingBlinks = 1;   // arm a double blink
				}
			}
		}
	}
	else
	{
		FaceDrive.EyeBlinkLeft  = FMath::FInterpTo(FaceDrive.EyeBlinkLeft, 0.f, DeltaTime, 20.f);
		FaceDrive.EyeBlinkRight = FMath::FInterpTo(FaceDrive.EyeBlinkRight, 0.f, DeltaTime, 20.f);
	}

	// -- Resolve the world point to look at (player's eyes, or an explicit actor) --
	const AActor* Owner = GetOwner();
	bHasGazeTarget = false;
	if (bGazeAtUser)
	{
		if (GazeTargetActor)
		{
			GazeTargetWorld = GazeTargetActor->GetActorLocation();
			bHasGazeTarget = true;
		}
		else if (const APlayerCameraManager* PCM = UGameplayStatics::GetPlayerCameraManager(this, 0))
		{
			GazeTargetWorld = PCM->GetCameraLocation();
			bHasGazeTarget = true;
		}
	}

	// -- Aim angles toward the target, measured in the mesh's own frame so they stay
	//    correct at any body rotation. The head turns for wide angles; the eyes cover
	//    the remainder and converge (vergence) on the exact point. --
	USkeletalMeshComponent* Mesh = ResolveFaceMesh();
	float TotalYaw = 0.f, TotalPitch = 0.f;      // head-origin aim
	float TotalYawL = 0.f, TotalYawR = 0.f;      // per-eye aim (vergence)
	float TotalPitchL = 0.f, TotalPitchR = 0.f;
	if (bHasGazeTarget && Owner)
	{
		const FTransform Frame = Mesh ? Mesh->GetComponentTransform() : Owner->GetActorTransform();
		auto AimFrom = [&](const FVector& FromWorld, float& OutYaw, float& OutPitch)
		{
			const FVector Local = Frame.InverseTransformVectorNoScale(GazeTargetWorld - FromWorld);
			if (!Local.IsNearlyZero())
			{
				OutYaw   = FMath::RadiansToDegrees(FMath::Atan2(Local.Y, Local.X));
				OutPitch = FMath::RadiansToDegrees(FMath::Atan2(Local.Z, FVector2D(Local.X, Local.Y).Size()));
				if (bInvertGazeYaw)   OutYaw = -OutYaw;
				if (bInvertGazePitch) OutPitch = -OutPitch;
				OutYaw   += GazeYawTrimDeg;
				OutPitch += GazePitchTrimDeg;
			}
		};

		FVector HeadWorld = Owner->GetActorLocation() + FVector(0.f, 0.f, EyeHeightCm);
		FVector LEyeWorld = HeadWorld, REyeWorld = HeadWorld;
		if (Mesh)
		{
			if (Mesh->DoesSocketExist(HeadBoneName))     HeadWorld = Mesh->GetSocketLocation(HeadBoneName);
			if (Mesh->DoesSocketExist(LeftEyeBoneName))  LEyeWorld = Mesh->GetSocketLocation(LeftEyeBoneName);
			if (Mesh->DoesSocketExist(RightEyeBoneName)) REyeWorld = Mesh->GetSocketLocation(RightEyeBoneName);
		}
		AimFrom(HeadWorld, TotalYaw, TotalPitch);
		if (bEyeVergence && Mesh)
		{
			AimFrom(LEyeWorld, TotalYawL, TotalPitchL);
			AimFrom(REyeWorld, TotalYawR, TotalPitchR);
		}
		else
		{
			TotalYawL = TotalYawR = TotalYaw;
			TotalPitchL = TotalPitchR = TotalPitch;
		}
	}

	// -- Head/neck: take a share of the aim, clamp to human limits, move slowly. --
	float HeadYawTarget = 0.f, HeadPitchTarget = 0.f;
	if (bHeadTracking)
	{
		HeadYawTarget   = FMath::Clamp(TotalYaw   * HeadFollowFraction, -HeadYawLimitDegrees,   HeadYawLimitDegrees);
		HeadPitchTarget = FMath::Clamp(TotalPitch * HeadFollowFraction, -HeadPitchLimitDegrees, HeadPitchLimitDegrees);
	}
	HeadYawVal   = FMath::FInterpTo(HeadYawVal,   HeadYawTarget,   DeltaTime, HeadTurnSpeed);
	HeadPitchVal = FMath::FInterpTo(HeadPitchVal, HeadPitchTarget, DeltaTime, HeadTurnSpeed);
	FaceDrive.HeadYaw   = HeadYawVal;
	FaceDrive.HeadPitch = HeadPitchVal;

	// -- Saccades: quick jumps to a fresh nearby target every so often -------
	SaccadeClock += DeltaTime;
	if (SaccadeClock >= SaccadeInterval)
	{
		SaccadeClock = 0.f;
		SaccadeInterval = FMath::FRandRange(0.6f, 2.4f);
		const float Dart = EyeDartDegrees * Scale;
		GazeTargetYaw   = FMath::FRandRange(-Dart, Dart);
		GazeTargetPitch = FMath::FRandRange(-Dart, Dart) * 0.6f;
		// While thinking, glance away from the user (natural gaze aversion);
		// while listening, hold steadier on them.
		if (AvatarState == EVcbAvatarState::Thinking)
		{
			GazeTargetYaw   += FMath::FRandRange(-1.f, 1.f) > 0.f ? 8.f : -8.f;
			GazeTargetPitch += 5.f;   // eyes drift up when recalling
		}
		else if (AvatarState == EVcbAvatarState::Listening)
		{
			GazeTargetYaw   *= 0.5f;
			GazeTargetPitch *= 0.5f;
		}
	}

	// Slow drift so the gaze never sits perfectly still between saccades.
	const float DriftYaw   = FMath::PerlinNoise1D(FaceNoiseTime * 0.23f + 31.f) * 1.5f * Scale;
	const float DriftPitch = FMath::PerlinNoise1D(FaceNoiseTime * 0.19f + 71.f) * 1.0f * Scale;

	const float Range = EyeGazeRangeDegrees;
	// Eyes cover the residual left after the head has turned, plus saccade + drift.
	const float ResYawL  = TotalYawL - HeadYawVal;
	const float ResYawR  = TotalYawR - HeadYawVal;
	const float ResPitch = ((TotalPitchL + TotalPitchR) * 0.5f) - HeadPitchVal;
	const float WantYawL  = FMath::Clamp(ResYawL  + GazeTargetYaw   + DriftYaw,   -Range, Range);
	const float WantYawR  = FMath::Clamp(ResYawR  + GazeTargetYaw   + DriftYaw,   -Range, Range);
	const float WantPitch = FMath::Clamp(ResPitch + GazeTargetPitch + DriftPitch, -Range, Range);
	// Eyes move fast (saccadic), so a snappy chase feels right.
	GazeYawLeft  = FMath::FInterpTo(GazeYawLeft,  WantYawL,  DeltaTime, 22.f);
	GazeYawRight = FMath::FInterpTo(GazeYawRight, WantYawR,  DeltaTime, 22.f);
	GazePitch    = FMath::FInterpTo(GazePitch,    WantPitch, DeltaTime, 22.f);
	FaceDrive.EyeYawLeft  = GazeYawLeft;
	FaceDrive.EyeYawRight = GazeYawRight;
	FaceDrive.EyeYaw   = (GazeYawLeft + GazeYawRight) * 0.5f;  // single-value back-compat
	FaceDrive.EyePitch = GazePitch;

	// -- Brows: subtle idle life + emotion (valence/arousal from the body path) --
	const float BrowNoise = 0.5f * (FMath::PerlinNoise1D(FaceNoiseTime * 0.15f + 5.f) + 1.f);
	const float RaiseTarget  = FMath::Clamp(
		0.05f + 0.25f * FMath::Max(0.f, BodyDrive.Valence)
		+ 0.15f * BodyDrive.Arousal + 0.05f * BrowNoise, 0.f, 1.f) * Scale;
	const float FurrowTarget = FMath::Clamp(
		0.30f * FMath::Max(0.f, -BodyDrive.Valence) + 0.10f * BodyDrive.Arousal, 0.f, 1.f) * Scale;
	BrowRaiseVal  = FMath::FInterpTo(BrowRaiseVal, RaiseTarget, DeltaTime, 4.f);
	BrowFurrowVal = FMath::FInterpTo(BrowFurrowVal, FurrowTarget, DeltaTime, 4.f);
	SquintVal     = FMath::FInterpTo(SquintVal,
		FMath::Clamp(0.15f * FurrowTarget + 0.10f * BodyDrive.Arousal, 0.f, 1.f), DeltaTime, 4.f);
	FaceDrive.BrowRaise  = BrowRaiseVal;
	FaceDrive.BrowFurrow = BrowFurrowVal;
	FaceDrive.Squint     = SquintVal;
}

// ---------------------------------------------------------------------------
// Audio2Face-3D streaming
// ---------------------------------------------------------------------------

bool UVcbAvatarComponent::InitAudio2Face()
{
	bA2FActive = false;
	A2FProvider = nullptr;
	CurveSource = nullptr;

	if (!bUseAudio2Face)
	{
		return false;
	}

	A2FProvider = IA2FProvider::FindProvider(A2FProviderName);
	if (A2FProvider == nullptr)
	{
		// Exact name not registered: prefer a local (on-GPU) provider, else the first available.
		const TArray<FName> Available = IA2FProvider::GetAvailableProviderNames();
		FName Chosen = NAME_None;
		for (const FName& N : Available)
		{
			if (N.ToString().Contains(TEXT("Local")))
			{
				Chosen = N;
				break;
			}
		}
		if (Chosen.IsNone() && Available.Num() > 0)
		{
			Chosen = Available[0];
		}
		if (!Chosen.IsNone())
		{
			A2FProvider = IA2FProvider::FindProvider(Chosen);
			if (A2FProvider != nullptr)
			{
				UE_LOG(LogVcbAvatarComp, Log,
					TEXT("Audio2Face-3D provider '%s' not found; using '%s' instead."),
					*A2FProviderName.ToString(), *Chosen.ToString());
				A2FProviderName = Chosen;
			}
		}
	}
	if (A2FProvider == nullptr)
	{
		TArray<FName> Available = IA2FProvider::GetAvailableProviderNames();
		FString AvailStr;
		for (const FName& N : Available)
		{
			AvailStr += (AvailStr.IsEmpty() ? TEXT("") : TEXT(", "));
			AvailStr += N.ToString();
		}
		UE_LOG(LogVcbAvatarComp, Warning,
			TEXT("No Audio2Face-3D provider available; using RMS jaw fallback. Available: [%s]"),
			*AvailStr);
		return false;
	}

	AActor* Owner = GetOwner();
	if (Owner == nullptr)
	{
		return false;
	}

	// Reuse a curve source placed in the Blueprint if there is one; otherwise spawn
	// it at runtime. The face AnimBP's "Apply ACE Animation" node auto-discovers it.
	CurveSource = Owner->FindComponentByClass<UACEAudioCurveSourceComponent>();
	if (CurveSource == nullptr)
	{
		CurveSource = NewObject<UACEAudioCurveSourceComponent>(Owner, TEXT("VcbACECurveSource"));
		if (CurveSource == nullptr)
		{
			return false;
		}
		CurveSource->RegisterComponent();
		if (USceneComponent* Root = Owner->GetRootComponent())
		{
			CurveSource->AttachToComponent(Root, FAttachmentTransformRules::KeepRelativeTransform);
		}
	}

	MinInitialSamples = FMath::Max(0, A2FProvider->GetMinimumInitialAudioSampleCount());
	A2FProvider->AllocateResources();   // warm up so the first utterance isn't slow
	bA2FActive = true;
	UE_LOG(LogVcbAvatarComp, Log, TEXT("Audio2Face-3D active via provider '%s' (min initial %d samples, ~%d ms)."),
		*A2FProviderName.ToString(), MinInitialSamples, (MinInitialSamples * 1000) / kA2FSampleRate);
	return true;
}

void UVcbAvatarComponent::BeginA2FStream(int32 SampleRate)
{
	if (!bA2FActive || A2FProvider == nullptr || CurveSource == nullptr || bA2FStreamOpen)
	{
		return;
	}
	if (SampleRate <= 0)
	{
		SampleRate = 24000;
	}
	CurrentSampleRate = SampleRate;
	ResetResampler();

	IA2FProvider::IA2FStream* Stream = A2FProvider->CreateA2FStream(CurveSource);
	if (Stream == nullptr)
	{
		UE_LOG(LogVcbAvatarComp, Warning, TEXT("Failed to create Audio2Face-3D stream."));
		return;
	}
	A2FStream = Stream;
	bA2FStreamOpen = true;
	bA2FEndPending = false;
	bSentInitialAudio = false;
	A2FSendStart = 0.0;
	A2FSamplesSent = 0;
	A2FOrigBytesSent = 0;
	PendingOriginal.Reset();

	// Feed the original (native-rate) audio through for playback so the ACE curve
	// source plays the same voice it's animating, in sync.
	if (IA2FPassthroughProvider* Passthrough = A2FProvider->GetAudioPassthroughProvider())
	{
		Passthrough->SetOriginalAudioParams(Stream, static_cast<uint32>(SampleRate), 1, 2);
	}

	// Prime the stream with silence: the model's warm-up frames land on silence
	// instead of the first word, and the extra silence on the playback side delays
	// the audible voice so the mouth doesn't lag it.
	const int32 PrimeMs = FMath::Max(0, A2FPrimeSilenceMs);
	const int32 OffsetMs = FMath::Max(0, AudioSyncOffsetMs);
	if (PrimeMs + OffsetMs > 0)
	{
		PendingOriginal.AddZeroed(((SampleRate * (PrimeMs + OffsetMs)) / 1000) * 2);   // PCM16 mono
		ResampleOut.AddZeroed((kA2FSampleRate * PrimeMs) / 1000);
	}
}

void UVcbAvatarComponent::PumpA2FAudio(const uint8* Pcm16Bytes, int32 NumBytes, int32 SampleRate)
{
	if (!bA2FStreamOpen || A2FStream == nullptr || Pcm16Bytes == nullptr || NumBytes < 2)
	{
		return;
	}
	IA2FProvider::IA2FStream* Stream = static_cast<IA2FProvider::IA2FStream*>(A2FStream);

	// Buffer the native-rate playback audio and resample the same span to 16 kHz for
	// inference. Both are released together in FlushA2FSamples so ACE receives the
	// playback chunk and its matching blendshape chunk paired 1:1 (keeps lip sync).
	PendingOriginal.Append(Pcm16Bytes, NumBytes);
	ResampleTo16k(reinterpret_cast<const int16*>(Pcm16Bytes), NumBytes / 2, SampleRate);
}

void UVcbAvatarComponent::FlushA2FSamples()
{
	if (!bA2FStreamOpen || A2FProvider == nullptr || A2FStream == nullptr)
	{
		return;
	}
	IA2FProvider::IA2FStream* Stream = static_cast<IA2FProvider::IA2FStream*>(A2FStream);

	if (ResampleOut.Num() > 0)
	{
		// NVIDIA guidance for local inference sharing the render GPU: send ~500 ms
		// up front, then feed near real time. Bursting whole replies makes A2F
		// infer seconds of audio at once and the game hitches/freezes mid-speech.
		const int32 InitialChunk = FMath::Max(kA2FSampleRate / 2, MinInitialSamples);
		int32 ToSend = 0;
		if (!bSentInitialAudio)
		{
			if (ResampleOut.Num() >= InitialChunk || bA2FEndPending)
			{
				ToSend = FMath::Min(ResampleOut.Num(), InitialChunk);
				A2FSendStart = FPlatformTime::Seconds();
			}
		}
		else
		{
			const double Elapsed = FPlatformTime::Seconds() - A2FSendStart;
			const int64 Budget =
				InitialChunk + static_cast<int64>(Elapsed * kA2FSampleRate) - A2FSamplesSent;
			// Cap per tick so a backlog after a long frame clears without a burst.
			ToSend = static_cast<int32>(FMath::Clamp<int64>(Budget, 0, kA2FSampleRate / 4));
			ToSend = FMath::Min(ToSend, ResampleOut.Num());
		}

		if (ToSend > 0)
		{
			// Enqueue the matching span of native-rate playback audio BEFORE sending
			// the inference chunk, as ACE requires, so voice and blendshapes advance
			// in lockstep.
			const int32 Rate = CurrentSampleRate > 0 ? CurrentSampleRate : 24000;
			const int64 OrigBytesTarget =
				((A2FSamplesSent + ToSend) * Rate / kA2FSampleRate) * 2;
			const int32 OrigBytes = static_cast<int32>(FMath::Clamp<int64>(
				OrigBytesTarget - A2FOrigBytesSent, 0, PendingOriginal.Num()));
			if (OrigBytes > 0)
			{
				if (IA2FPassthroughProvider* Passthrough = A2FProvider->GetAudioPassthroughProvider())
				{
					Passthrough->EnqueueOriginalSamples(
						Stream, TArrayView<const uint8>(PendingOriginal.GetData(), OrigBytes));
				}
				PendingOriginal.RemoveAt(0, OrigBytes, EAllowShrinking::No);
				A2FOrigBytesSent += OrigBytes;
			}
			A2FProvider->SendAudioSamples(
				Stream,
				TArrayView<const int16>(ResampleOut.GetData(), ToSend),
				TOptional<FAudio2FaceEmotion>(),
				nullptr);
			ResampleOut.RemoveAt(0, ToSend, EAllowShrinking::No);
			A2FSamplesSent += ToSend;
			bSentInitialAudio = true;
		}
	}

	if (bA2FEndPending && ResampleOut.Num() == 0)
	{
		// Everything (including the end pad) has been sent: release any rounding
		// remainder of playback audio and close. The model holds less than one
		// inference block now, so the blocking end call is cheap, and playback
		// still trails the inference lead - no underrun on the last word.
		if (PendingOriginal.Num() > 0)
		{
			if (IA2FPassthroughProvider* Passthrough = A2FProvider->GetAudioPassthroughProvider())
			{
				Passthrough->EnqueueOriginalSamples(
					Stream, TArrayView<const uint8>(PendingOriginal.GetData(), PendingOriginal.Num()));
			}
			PendingOriginal.Reset();
		}
		EndA2FStream();
	}
}

void UVcbAvatarComponent::RequestA2FStreamEnd()
{
	if (!bA2FStreamOpen || bA2FEndPending)
	{
		return;
	}
	// Pad a short silence tail so the final syllable always has buffered audio
	// queued behind it (prevents an end-of-buffer underrun that glitches the last
	// word). The inference side gets an extra AudioSyncOffsetMs so the offset-
	// delayed playback audio is fully released before the stream closes.
	const int32 Rate = CurrentSampleRate > 0 ? CurrentSampleRate : 24000;
	PendingOriginal.AddZeroed((Rate / 5) * 2);   // 200 ms, PCM16 mono
	ResampleOut.AddZeroed((kA2FSampleRate * (200 + FMath::Max(0, AudioSyncOffsetMs))) / 1000);
	bA2FEndPending = true;
}

void UVcbAvatarComponent::EndA2FStream()
{
	if (bA2FStreamOpen && A2FProvider != nullptr && A2FStream != nullptr)
	{
		A2FProvider->EndOutgoingStream(static_cast<IA2FProvider::IA2FStream*>(A2FStream));
	}
	A2FStream = nullptr;
	bA2FStreamOpen = false;
	bA2FEndPending = false;
	PendingOriginal.Reset();
	ResetResampler();
}

void UVcbAvatarComponent::ResampleTo16k(const int16* In, int32 NumSamples, int32 SrcRate)
{
	if (In == nullptr || NumSamples <= 0)
	{
		return;
	}
	if (SrcRate == kA2FSampleRate)
	{
		ResampleOut.Append(In, NumSamples);
		ResamplePrev = In[NumSamples - 1];
		return;
	}

	const double Step = static_cast<double>(SrcRate) / static_cast<double>(kA2FSampleRate);
	double Pos = ResamplePos;   // in source-sample units, relative to In[0]; may start negative
	while (true)
	{
		const int32 I0 = static_cast<int32>(FMath::FloorToDouble(Pos));
		if (I0 + 1 >= NumSamples)
		{
			break;   // need the next buffer to interpolate the upper sample
		}
		const int16 S0 = (I0 < 0) ? ResamplePrev : In[I0];
		const int16 S1 = In[I0 + 1];
		const double Frac = Pos - static_cast<double>(I0);
		const int32 V = FMath::RoundToInt(static_cast<double>(S0) +
			(static_cast<double>(S1) - static_cast<double>(S0)) * Frac);
		ResampleOut.Add(static_cast<int16>(FMath::Clamp(V, -32768, 32767)));
		Pos += Step;
	}
	ResamplePrev = In[NumSamples - 1];
	ResamplePos = Pos - static_cast<double>(NumSamples);   // carry into the next buffer
}

void UVcbAvatarComponent::ResetResampler()
{
	ResampleOut.Reset();
	ResamplePos = 0.0;
	ResamplePrev = 0;
	bSentInitialAudio = false;
}
