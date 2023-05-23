

vector g_vViewport;
vector g_vCameraParams;
matrix g_mInvViewProj;
vector g_vEye;

// TRESSFX Binding =========================================

vector vFragmentBufferSize;
int nNodePoolSize;

//Texture2D tFragmentListHead;
//StructuredBuffer LinkedListSRV;

// =========================================



//#include "SuGamma.shl"
//#include "iHairShading.shl"
#include "TressFXRendering.hlsl"
#include "TressFXFPPLL_Resolve.hlsl"

//#define SU_LINEAR_SPACE_LIGHTING
//#include "SuLighting.shl"
//#include "iBRDF.shl"
//#include "iShadows.shl"
//#include "iShadowedLighting.shl"
//#include "iLighting.shl"

//#include "iCommon.shl"
//#include "iShortCut.shl"
//#include "TressFXShortCut.hlsl"

struct VS_OUTPUT_SCREENQUAD
{
	float4 vPosition : SV_POSITION;
	float2 vTex      : TEXCOORD;
};

static const float2 Positions[] = { {-1, -1}, {1, -1}, {-1,1}, {1,1} };

VS_OUTPUT_SCREENQUAD main(uint gl_VertexID : SV_VertexID)
{
	VS_OUTPUT_SCREENQUAD output;
	//gl_Position = vec4( vPosition.xy, 0.5, 1);
	output.vPosition = float4(Positions[gl_VertexID].xy, 0, 1);

	// y down.
	output.vTex = float2(Positions[gl_VertexID].x, -Positions[gl_VertexID].y) * 0.5 + 0.5;
	return output;
}

technique11 TressFX2
{
	pass P0
	{
		SetVertexShader(CompileShader(vs_5_0, main()));
	}
}