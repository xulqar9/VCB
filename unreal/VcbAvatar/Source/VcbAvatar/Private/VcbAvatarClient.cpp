#include "VcbAvatarClient.h"

#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "HAL/RunnableThread.h"
#include "HAL/PlatformProcess.h"
#include "Misc/ScopeLock.h"

DEFINE_LOG_CATEGORY_STATIC(LogVcbAvatar, Log, All);

FVcbAvatarClient::FVcbAvatarClient(const FString& InHost, int32 InPort,
                                   TQueue<FVcbAvatarMessage, EQueueMode::Spsc>* InQueue)
	: Host(InHost)
	, Port(InPort)
	, Queue(InQueue)
{
}

FVcbAvatarClient::~FVcbAvatarClient()
{
	StopThread();
}

void FVcbAvatarClient::StartThread()
{
	if (Thread == nullptr)
	{
		bStop = false;
		Thread = FRunnableThread::Create(this, TEXT("VcbAvatarClient"), 0, TPri_Normal);
	}
}

void FVcbAvatarClient::StopThread()
{
	bStop = true;
	ShutdownSocket();         // unblocks a Recv/connect in flight (does NOT destroy)
	if (Thread != nullptr)
	{
		Thread->Kill(true);   // waits for Run() to return; it destroys the socket
		delete Thread;
		Thread = nullptr;
	}
	CloseSocket();            // safety net; idempotent if the thread already freed it
}

bool FVcbAvatarClient::Init()
{
	return true;
}

void FVcbAvatarClient::Stop()
{
	bStop = true;
	ShutdownSocket();
}

void FVcbAvatarClient::Exit()
{
}

bool FVcbAvatarClient::Connect()
{
	ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (SS == nullptr)
	{
		return false;
	}

	TSharedRef<FInternetAddr> Addr = SS->CreateInternetAddr();
	bool bValidIp = false;
	Addr->SetIp(*Host, bValidIp);
	if (!bValidIp)
	{
		// Resolve a hostname (e.g. "localhost") if a literal IP wasn't given.
		FAddressInfoResult Info = SS->GetAddressInfo(*Host, nullptr, EAddressInfoFlags::Default, NAME_None);
		if (Info.ReturnCode != SE_NO_ERROR || Info.Results.Num() == 0)
		{
			return false;
		}
		Addr = Info.Results[0].Address->Clone();
	}
	Addr->SetPort(Port);

	FSocket* NewSocket = SS->CreateSocket(NAME_Stream, TEXT("VcbAvatarSocket"), Addr->GetProtocolType());
	if (NewSocket == nullptr)
	{
		return false;
	}
	{
		FScopeLock Lock(&SocketCS);
		Socket = NewSocket;
	}

	Socket->SetNonBlocking(false);
	Socket->SetNoDelay(true);
	if (!Socket->Connect(*Addr))
	{
		CloseSocket();
		return false;
	}
	UE_LOG(LogVcbAvatar, Log, TEXT("connected to VCB bridge %s:%d"), *Host, Port);
	return true;
}

void FVcbAvatarClient::CloseSocket()
{
	FScopeLock Lock(&SocketCS);
	if (Socket != nullptr)
	{
		Socket->Close();
		if (ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM))
		{
			SS->DestroySocket(Socket);
		}
		Socket = nullptr;
	}
}

void FVcbAvatarClient::ShutdownSocket()
{
	// Only close the connection so a blocking Recv/connect returns; the socket
	// thread destroys the FSocket itself in Run(). Never free it from here.
	FScopeLock Lock(&SocketCS);
	if (Socket != nullptr)
	{
		Socket->Close();
	}
}

bool FVcbAvatarClient::RecvAll(uint8* Buf, int32 Num)
{
	int32 Got = 0;
	while (Got < Num && !bStop)
	{
		if (Socket == nullptr)
		{
			return false;
		}
		// Wake up periodically so a Stop() request is honoured quickly.
		if (!Socket->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromMilliseconds(100)))
		{
			continue;
		}
		int32 Read = 0;
		const bool bOk = Socket->Recv(Buf + Got, Num - Got, Read);
		if (!bOk || Read <= 0)
		{
			return false;   // peer closed or error
		}
		Got += Read;
	}
	return Got == Num;
}

void FVcbAvatarClient::EnqueueSimple(EVcbMsgType InType)
{
	FVcbAvatarMessage Msg;
	Msg.Type = InType;
	Queue->Enqueue(MoveTemp(Msg));
}

uint32 FVcbAvatarClient::Run()
{
	uint8 Header[5];
	while (!bStop)
	{
		if (!Connect())
		{
			FPlatformProcess::Sleep(1.0f);
			continue;
		}
		EnqueueSimple(EVcbMsgType::Connected);

		while (!bStop)
		{
			if (!RecvAll(Header, 5))
			{
				break;
			}
			const uint8 WireType = Header[0];
			const uint32 Len =
				(uint32(Header[1]) << 24) | (uint32(Header[2]) << 16) |
				(uint32(Header[3]) << 8) | uint32(Header[4]);

			TArray<uint8> Payload;
			if (Len > 0)
			{
				Payload.SetNumUninitialized(static_cast<int32>(Len));
				if (!RecvAll(Payload.GetData(), static_cast<int32>(Len)))
				{
					break;
				}
			}

			FVcbAvatarMessage Msg;
			if (WireType == 0)
			{
				Msg.Type = EVcbMsgType::Json;
				FUTF8ToTCHAR Conv(reinterpret_cast<const ANSICHAR*>(Payload.GetData()), Payload.Num());
				Msg.Json = FString(Conv.Length(), Conv.Get());
			}
			else if (WireType == 1)
			{
				Msg.Type = EVcbMsgType::Audio;
				const int32 NumSamples = Payload.Num() / 4;   // float32 -> samples
				if (NumSamples > 0)
				{
					const float* In = reinterpret_cast<const float*>(Payload.GetData());
					Msg.Pcm16.SetNumUninitialized(NumSamples * 2);
					int16* Out = reinterpret_cast<int16*>(Msg.Pcm16.GetData());
					double SumSq = 0.0;
					for (int32 i = 0; i < NumSamples; ++i)
					{
						const float S = In[i];
						SumSq += double(S) * double(S);
						Out[i] = static_cast<int16>(FMath::Clamp(S * 32767.f, -32768.f, 32767.f));
					}
					Msg.Rms = static_cast<float>(FMath::Sqrt(SumSq / NumSamples));
				}
			}
			else
			{
				continue;   // unknown wire type, ignore
			}
			Queue->Enqueue(MoveTemp(Msg));
		}

		CloseSocket();
		EnqueueSimple(EVcbMsgType::Disconnected);
		if (!bStop)
		{
			FPlatformProcess::Sleep(1.0f);   // backoff before reconnecting
		}
	}
	return 0;
}
