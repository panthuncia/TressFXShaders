


//=================================================================================================================================
//
// Author: Karl Hillesland (karl.hillesland@amd.com)
//         AMD, Inc.
//
//=================================================================================================================================
// $Id: //depot/3darg/Demos/Effects/TressFXRelease/amd_tressfx_sample/bin/Effects/D3D11/oHair.sufx#3 $ 
// 
// Last check-in:  $DateTime: 2017/03/31 12:58:07 $ 
// Last edited by: $Author: khillesl $
//=================================================================================================================================
//   (C) AMD, Inc. 2013 All rights reserved. 
//=================================================================================================================================


// ART NOTES ======================================================================================================================
//hair shader

// INCLUDES ======================================================================================================================

// #include "SuMath.shl"
// #include "SuGamma.shl"

// #include "iHairShading.shl"
#include "TressFXFPPLL_Common.hlsl"
// #include "TressFXRendering.hlsl"
// #include "TressFXStrands.hlsl"
// #include "iCommon.shl"
// #include "iShortCut.shl"
// #include "TressFXShortCut.hlsl"

// #define SU_LINEAR_SPACE_LIGHTING
// #include "SuLighting.shl"
// #include "iBRDF.shl"
// #include "iShadows.shl"
// #include "iShadowedLighting.shl"
// #include "iLighting.shl"


//Matrix g_mVP < AppUpdate = ViewProjMatrix >
//Matrix g_mInvViewProj < AppUpdate = ViewProjMatrixInverse >
//Matrix g_mView < AppUpdate = ViewMatrix >
//Matrix g_mProjection < AppUpdate = ProjMatrix >
//Vector g_vEye < AppUpdate = CameraPosition >
//Vector g_vViewport < AppUpdate = Viewport >
//Vector g_vCameraParams < AppUpdate = CameraProjParams >

//Bool g_bThinTip < AppUpdate = ScriptVariable, Name = "g_bThinTip" >
//Float g_Ratio < AppUpdate = ScriptVariable, Name = "g_Ratio" >

//Float zNear < AppUpdate = ScriptVariable, Name = "DOF.zNear" >
//Float zFar  < AppUpdate = ScriptVariable, Name = "DOF.zFar" >
//Float focusDistance < AppUpdate = ScriptVariable, Name = "DOF.focusDistance" >
//Float fStop < AppUpdate = ScriptVariable, Name = "DOF.fStop" >
//Float focalLength < AppUpdate = ScriptVariable, Name = "DOF.focalLength" >
//Float cocHairThreshold <AppUpdate = ScriptVariable, Name = "DOF.cocHairThreshold" >

matrix g_mVP;
matrix g_mInvViewProj;
matrix g_mView;
matrix g_mProjection;
vector g_vEye;
vector g_vViewport;
vector g_vCameraParams;
bool g_bThinTip;
float g_Ratio;
float zNear;
float zFar;
float focusDistance;
float fStop;
float focalLength;
float cocHairThreshold;

// TRESSFX Binding =========================================

StructuredBuffer<float4> g_GuideHairVertexPositions;
StructuredBuffer<float4> g_GuideHairVertexTangents;

StructuredBuffer<float> g_HairThicknessCoeffs;

StructuredBuffer<float2> g_HairStrandTexCd;

vector vFragmentBufferSize;
// =========================================
			//float4x4 g_mInvViewProj;
			//float4 g_vViewport;


//#include "TressFXFPPLL_Common.hlsl"
#include "TressFXFPPLL_Build.hlsl"

#include "GetWorldPos.hlsl"
#include "TressFXRendering.hlsl"
#include "ErrorChecking.hlsl"


struct PS_INPUT_HAIR_AA
{
    float4 Position : SV_POSITION;
    float4 Tangent : Tangent;
    float4 p0p1 : TEXCOORD0;
    float3 strandColor : TEXCOORD1;
				//float4 PosCheck : POSCHECK;
};



[earlydepthstencil]
VOID_RETURN main(PS_INPUT_HAIR_AA input) VOID_RETURN_SEMANTIC
{
    float3 vNDC = ScreenToNDC(input.Position.xyz, g_vViewport);
    float3 vPositionWS = NDCToWorld(vNDC, g_mInvViewProj);

    float coverage = ComputeCoverage(input.p0p1.xy, input.p0p1.zw, vNDC.xy, g_vViewport.zw);
    float alpha = coverage * g_MatBaseColor.a;

				ASSERT(coverage >= 0) 
    if (alpha < 1.0 / 255.0)
					RETURN_NOTHING

    int2 vScreenAddress = int2(input.Position.xy);
			// Allocate a new fragment
    int nNewFragmentAddress = AllocateFragment(vScreenAddress);
			ASSERT ( nNewFragmentAddress != FRAGMENT_LIST_NULL )

    int nOldFragmentAddress = MakeFragmentLink(vScreenAddress, nNewFragmentAddress);
    WriteFragmentAttributes(nNewFragmentAddress, nOldFragmentAddress, float4(input.Tangent.xyz * 0.5 + float3(0.5, 0.5, 0.5), alpha), input.strandColor.xyz, input.Position.z);

    RETURN_NOTHING
}
technique11 TressFX2
{
	pass P0
	{
        SetPixelShader(CompileShader(ps_5_0, main()));
    }
}