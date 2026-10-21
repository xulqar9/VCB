#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/ThreadSafeBool.h"
#include "Containers/Queue.h"

class FSocket;
class FRunnableThread;

/** Message kinds handed from the socket thread to the game thread. */
enum class EVcbMsgType : uint8
{
	Json = 0,          // control JSON from the Python bridge (matches wire type 0)
	Audio = 1,         // decoded int16 PCM ready to queue (wire type 1 was float32)
	Connected = 10,    // synthetic: socket connected
	Disconnected = 11  // synthetic: socket dropped
};

/** One parsed frame. Audio is already converted to little-endian int16 PCM. */
struct FVcbAvatarMessage
{
	EVcbMsgType Type = EVcbMsgType::Json;
	FString Json;
	TArray<uint8> Pcm16;   // int16 mono PCM (what USoundWaveProcedural wants)
	float Rms = 0.f;       // frame loudness 0..1, a cheap mouth-open signal
};

/**
 * Background thread that connects to the VCB avatar bridge (a local TCP server),
 * reads length-prefixed frames, converts audio to int16 PCM, and pushes results
 * onto an SPSC queue drained by UVcbAvatarComponent on the game thread.
 *
 * Wire format (see vcb/avatar/bridge.py): 5-byte header = [uint8 type][uint32 BE
 * length] then payload. type 0 = UTF-8 JSON, type 1 = float32 LE mono audio.
 * Reconnects automatically with a short backoff until Stop() is called.
 */
class FVcbAvatarClient : public FRunnable
{
public:
	FVcbAvatarClient(const FString& InHost, int32 InPort,
	                 TQueue<FVcbAvatarMessage, EQueueMode::Spsc>* InQueue);
	virtual ~FVcbAvatarClient() override;

	void StartThread();
	void StopThread();

	// FRunnable
	virtual bool Init() override;
	virtual uint32 Run() override;
	virtual void Stop() override;
	virtual void Exit() override;

private:
	bool Connect();
	void CloseSocket();
	/** Close the connection (to unblock a pending Recv/connect) WITHOUT destroying
	 *  the FSocket; the socket thread owns destruction, so this avoids a double-free. */
	void ShutdownSocket();
	/** Block until exactly Num bytes are read, bailing out promptly on Stop/EOF. */
	bool RecvAll(uint8* Buf, int32 Num);
	void EnqueueSimple(EVcbMsgType InType);

	FString Host;
	int32 Port;
	TQueue<FVcbAvatarMessage, EQueueMode::Spsc>* Queue;

	FSocket* Socket = nullptr;
	FRunnableThread* Thread = nullptr;
	FThreadSafeBool bStop = false;
	FCriticalSection SocketCS;   // serialises socket close/destroy across threads
};
