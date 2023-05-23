// ART NOTES ======================================================================================================================

// DEFINES =======================================================================================================================
#define THREAD_GROUP_SIZE 64

// INCLUDES =======================================================================================================================
#include "TressFXSDFCollision.hlsl"
#include "iTressFXBoneSkinning.hlsl"

//-----------------------
// Collision mesh
//-----------------------

//// constant buffer
//matrix g_BoneSkinningMatrix[AMD_TRESSFX_MAX_NUM_BONES]
//int g_NumMeshVertices
//
//// UAVs
//RWStructuredBuffer collMeshVertexPositions < >
//
//// SRVs
//StructuredBuffer g_BoneSkinningData < >
//StructuredBuffer initialVertexPositions < >

//--------------
// SDF
//--------------
#define INITIAL_DISTANCE 1e10f
#define MARGIN g_CellSize
#define GRID_MARGIN int3(1, 1, 1)

//Matrix g_ModelTransformForHead < >
//Matrix g_ModelInvTransformForHead < >


////Actually contains floats; make sure to use asfloat() when accessing. uint is used to allow atomics.
//RWStructuredBuffer g_SignedDistanceField < >
//RWStructuredBuffer g_Sign < >
//RWStructuredBuffer g_HairVertices < >
//RWStructuredBuffer g_PrevHairVertices < >
//RWStructuredBuffer g_PointsSDFDebugUAV < >
//
////Triangle input to SDF builder
//StructuredBuffer g_TrimeshVertices < >
//StructuredBuffer g_TrimeshVertexIndices < >

technique11 TressFX_BoneSkinning
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, BoneSkinning()));
	}
}


technique11 TressFXSimulation_InitializeSignedDistanceField
{
	pass C0
	{
		SetComputeShader(CompileShader(cs_5_0, InitializeSignedDistanceField()));
	}
}

technique11 TressFXSimulation_ConstructSignedDistanceField
	{
		pass C0
		{
			SetComputeShader(CompileShader(cs_5_0, ConstructSignedDistanceField()));
	}
	}

technique11 TressFXSimulation_FinalizeSignedDistanceField
	{
		pass C0
		{
			SetComputeShader(CompileShader(cs_5_0, FinalizeSignedDistanceField()));
	}
	}

technique11 TressFXSimulation_CollideHairVerticesWithSdf_forward
	{
		pass C0
		{
			SetComputeShader(CompileShader(cs_5_0, CollideHairVerticesWithSdf_forward()));
	}
	}

technique11 TressFXSimulation_CollideHairVerticesWithSdf
	{
		pass C0
		{
			SetComputeShader(CompileShader(cs_5_0, CollideHairVerticesWithSdf()));
	}
}


