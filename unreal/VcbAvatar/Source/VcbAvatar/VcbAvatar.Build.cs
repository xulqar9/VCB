using UnrealBuildTool;

public class VcbAvatar : ModuleRules
{
	public VcbAvatar(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Sockets",
			"Networking",
			"Json"
		});

		// NVIDIA ACE / Audio2Face-3D: stream mic-driven TTS audio into the local
		// A2F provider and let it drive the MetaHuman face. ACECore has the provider
		// interface + anim-data types; ACERuntime has the audio curve source component.
		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"ACECore",
			"ACERuntime"
		});
	}
}
