// For capsule collisions
#define TRESSFX_MAX_NUM_COLLISION_CAPSULES 8

// As a reference, following variables are defined in oPhongSk.sufx
//#define SU_SKINNING_MAX_BONES 200
//#define SU_SKINNING_NUM_BONES 4

// INCLUDES =======================================================================================================================
#include "TressFXSimulation.hlsl"


technique11 TressFXSimulation_IntegrationAndGlobalShapeConstraints
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, IntegrationAndGlobalShapeConstraints()));
	}
}

technique11 TressFXSimulation_VelocityShockPropagation
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, VelocityShockPropagation()));
	}
}

technique11 TressFXSimulation_LocalShapeConstraints
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, LocalShapeConstraints()));
	}
}

technique11 TressFXSimulation_LengthConstriantsWindAndCollision
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, LengthConstriantsWindAndCollision()));
	}
}

technique11 TressFXSimulation_UpdateFollowHairVertices
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, LengthConstriantsWindAndCollision()));
	}
}