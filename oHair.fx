#line 109 "oHair.sufx"



#line 31 "../../amd_tressfx/src/shaders/TressFXRendering.hlsl"



#define AMD_PI 3.1415926
#define AMD_e 2.71828183

#define AMD_TRESSFX_KERNEL_SIZE 5

#ifndef FRAGMENT_LIST_NULL
#define FRAGMENT_LIST_NULL 0xffffffff
#endif

#define HAS_COLOR 1

// We might break this down further.
cbuffer tressfxShadeParameters
{
    //float       g_FiberAlpha        ; // Is this redundant with g_MatBaseColor.a?
    float       g_HairShadowAlpha;
    float       g_FiberRadius;
    float       g_FiberSpacing;

    float4      g_MatBaseColor;
    float4      g_MatKValue;

    float       g_fHairKs2;
    float       g_fHairEx2;
    int g_NumVerticesPerStrand;

    //float padding_ 
}


struct HairShadeParams
{
    float3 cColor;
    float fRadius;
    float fSpacing;
    float fAlpha;
};

float4x4 g_mInvViewProj;

struct PPLL_STRUCT
{
    uint	depth;
    uint	data;
    uint    color;
    uint    uNext;
};


RWTexture2D<int>    tRWFragmentListHead;

RWStructuredBuffer<PPLL_STRUCT> LinkedListUAV;

int nNodePoolSize;

// fDepthDistanceWS is the world space distance between the point on the surface and the point in the shadow map.
// fFiberSpacing is the assumed, average, world space distance between hair fibers.
// fFiberRadius in the assumed average hair radius.
// fHairAlpha is the alpha value for the hair (in terms of translucency, not coverage.)
// Output is a number between 0 (totally shadowed) and 1 (lets everything through)
float ComputeShadowAttenuation(float fDepthDistanceWS, float fFiberSpacing, float fFiberRadius, float fHairAlpha)
{
    float numFibers = fDepthDistanceWS / (fFiberSpacing * fFiberRadius);

    // if occluded by hair, there is at least one fiber
    [flatten] if (fDepthDistanceWS > 1e-5)
        numFibers = max(numFibers, 1);
    return pow(abs(1 - fHairAlpha), numFibers);
}

float ComputeShadowAttenuation(float fDepthDistanceWS, HairShadeParams params)
{
    return ComputeShadowAttenuation(fDepthDistanceWS, params.fSpacing, params.fRadius, params.fAlpha);
}


float ComputeHairShadowAttenuation(float fDepthDistanceWS, float fFiberSpacing, float fFiberRadius, float fHairAlpha)
{
    float numFibers = fDepthDistanceWS / (fFiberSpacing * fFiberRadius);

    fHairAlpha *= 0.5;

    // if occluded by hair, there is at least one fiber
    [flatten] if (fDepthDistanceWS > 1e-5)
        numFibers = max(numFibers, 1);
    return pow(abs(1 - fHairAlpha), numFibers);
}


float ComputeHairShadowAttenuation(float fDepthDistanceWS)
{
    return ComputeHairShadowAttenuation(fDepthDistanceWS, g_FiberRadius, g_FiberSpacing, g_HairShadowAlpha);
}


// Returns a float3 which is the scale for diffuse, spec term, and colored spec term.
//
// The diffuse term is from Kajiya.
// 
// The spec term is what Marschner calls "R", reflecting directly off the surface of the hair, 
// taking the color of the light like a dielectric specular term.  This highlight is shifted 
// towards the root of the hair.
//
// The colored spec term is caused by light passing through the hair, bouncing off the back, and 
// coming back out.  It therefore picks up the color of the light.  
// Marschner refers to this term as the "TRT" term.  This highlight is shifted towards the 
// tip of the hair.
//
// vEyeDir, vLightDir and vTangentDir are all pointing out.
// coneAngleRadians explained below.
//
// 
// hair has a tiled-conical shape along its lenght.  Sort of like the following.
// 
// \    /
//  \  /
// \    /
//  \  /  
//
// The angle of the cone is the last argument, in radians.  
// It's typically in the range of 5 to 10 degrees
float3 TressFX_ComputeDiffuseSpecFactors(float3 vEyeDir, float3 vLightDir, float3 vTangentDir, float coneAngleRadians = 10 * AMD_PI / 180)
{

    // From old code.
    float Kd = g_MatKValue.y;
    float Ks1 = g_MatKValue.z;
    float Ex1 = g_MatKValue.w;
    float Ks2 = g_fHairKs2;
    float Ex2 = g_fHairEx2;

    // in Kajiya's model: diffuse component: sin(t, l)
    float cosTL = (dot(vTangentDir, vLightDir));
    float sinTL = sqrt(1 - cosTL * cosTL);
    float diffuse = sinTL; // here sinTL is apparently larger than 0

    float cosTRL = -cosTL;
    float sinTRL = sinTL;
    float cosTE = (dot(vTangentDir, vEyeDir));
    float sinTE = sqrt(1 - cosTE * cosTE);

    // primary highlight: reflected direction shift towards root (2 * coneAngleRadians)
    float cosTRL_root = cosTRL * cos(2 * coneAngleRadians) - sinTRL * sin(2 * coneAngleRadians);
    float sinTRL_root = sqrt(1 - cosTRL_root * cosTRL_root);
    float specular_root = max(0, cosTRL_root * cosTE + sinTRL_root * sinTE);

    // secondary highlight: reflected direction shifted toward tip (3*coneAngleRadians)
    float cosTRL_tip = cosTRL * cos(-3 * coneAngleRadians) - sinTRL * sin(-3 * coneAngleRadians);
    float sinTRL_tip = sqrt(1 - cosTRL_tip * cosTRL_tip);
    float specular_tip = max(0, cosTRL_tip * cosTE + sinTRL_tip * sinTE);

    return float3(Kd * diffuse, Ks1 * pow(specular_root, Ex1), Ks2 * pow(specular_tip, Ex2));
}


//--------------------------------------------------------------------------------------
// ComputeCoverage
//
// Calculate the pixel coverage of a hair strand by computing the hair width
//--------------------------------------------------------------------------------------
float ComputeCoverage(float2 p0, float2 p1, float2 pixelLoc, float2 winSize)
{
    // p0, p1, pixelLoc are in d3d clip space (-1 to 1)x(-1 to 1)

    // Scale positions so 1.f = half pixel width
    p0 *= winSize;
    p1 *= winSize;
    pixelLoc *= winSize;

    float p0dist = length(p0 - pixelLoc);
    float p1dist = length(p1 - pixelLoc);
    float hairWidth = length(p0 - p1);

    // will be 1.f if pixel outside hair, 0.f if pixel inside hair
    float outside = any(float2(step(hairWidth, p0dist), step(hairWidth, p1dist)));

    // if outside, set sign to -1, else set sign to 1
    float sign = outside > 0.f ? -1.f : 1.f;

    // signed distance (positive if inside hair, negative if outside hair)
    float relDist = sign * saturate(min(p0dist, p1dist));

    // returns coverage based on the relative distance
    // 0, if completely outside hair edge
    // 1, if completely inside hair edge
    return (relDist + 1.f) * 0.5f;
}







#line 111 "oHair.sufx"


#line 31 "../../amd_tressfx/src/shaders/TressFXStrands.hlsl"

#define TRESSFX_FLOAT_EPSILON 1e-7


static const bool g_bExpandPixels = false;
bool g_bThinTip;
float g_Ratio;


texture2D<float3> g_txHairColor;
sampler g_samLinearWrap;



//--------------------------------------------------------------------------------------
//
//     Controls whether you do mul(M,v) or mul(v,M)
//     i.e., row major vs column major
//
//--------------------------------------------------------------------------------------
float4 MatrixMult(float4x4 m, float4 v)
{
    return mul(m, v);
}



//--------------------------------------------------------------------------------------
//
//      Safe_normalize-float2
//
//--------------------------------------------------------------------------------------
float2 Safe_normalize(float2 vec)
{
    float len = length(vec);
    return len >= TRESSFX_FLOAT_EPSILON ? (vec * rcp(len)) : float2(0, 0);
}

//--------------------------------------------------------------------------------------
//
//      Safe_normalize-float3
//
//--------------------------------------------------------------------------------------
float3 Safe_normalize(float3 vec)
{
    float len = length(vec);
    return len >= TRESSFX_FLOAT_EPSILON ? (vec * rcp(len)) : float3(0, 0, 0);
}

//--------------------------------------------------------------------------------------
//
//      Structured buffers with hair vertex info
//      Accessors to allow changes to how they are accessed.
//
//--------------------------------------------------------------------------------------
StructuredBuffer<float4> g_GuideHairVertexPositions;
StructuredBuffer<float4> g_GuideHairVertexTangents;
StructuredBuffer<float> g_HairThicknessCoeffs;
StructuredBuffer<float2> g_HairStrandTexCd;

inline float4 GetPosition(int index)
{
    return g_GuideHairVertexPositions[index];
}
inline float4 GetTangent(int index)
{
    return g_GuideHairVertexTangents[index];
}
inline float GetThickness(int index)
{
    return g_HairThicknessCoeffs[index];
}


float3 GetStrandColor(int index)
{
    float2 texCd = g_HairStrandTexCd[(uint)index / (uint)g_NumVerticesPerStrand];
    float3 color = g_txHairColor.SampleLevel(g_samLinearWrap, texCd, 0).xyz;// * g_MatBaseColor.xyz;
    return (color);
}

struct TressFXVertex
{
    float4 Position;
    float4 Tangent;
    float4 p0p1;
    float3 strandColor;
};

TressFXVertex GetExpandedTressFXVert(uint vertexId, float3 eye, float2 winSize, float4x4 viewProj)
{

    // Access the current line segment
    uint index = vertexId / 2;  // vertexId is actually the indexed vertex id when indexed triangles are used

    // Get updated positions and tangents from simulation result
    float3 v = g_GuideHairVertexPositions[index].xyz;
    float3 t = g_GuideHairVertexTangents[index].xyz;

    // Get hair strand thickness
    float ratio = (g_bThinTip > 0) ? g_Ratio : 1.0;

    // Calculate right and projected right vectors
    float3 right = Safe_normalize(cross(t, Safe_normalize(v - eye)));
    float2 proj_right = Safe_normalize(MatrixMult(viewProj, float4(right, 0)).xy);

    // g_bExpandPixels should be set to 0 at minimum from the CPU side; this would avoid the below test
    float expandPixels = (g_bExpandPixels < 0) ? 0.0 : 0.71;

    // Calculate the negative and positive offset screenspace positions
    float4 hairEdgePositions[2]; // 0 is negative, 1 is positive
    hairEdgePositions[0] = float4(v + -1.0 * right * ratio * g_FiberRadius, 1.0);
    hairEdgePositions[1] = float4(v + 1.0 * right * ratio * g_FiberRadius, 1.0);
    hairEdgePositions[0] = MatrixMult(viewProj, hairEdgePositions[0]);
    hairEdgePositions[1] = MatrixMult(viewProj, hairEdgePositions[1]);

    // Write output data
    TressFXVertex Output = (TressFXVertex)0;
    float fDirIndex = (vertexId & 0x01) ? -1.0 : 1.0;
    Output.Position = ((vertexId & 0x01) ? hairEdgePositions[0] : hairEdgePositions[1]) + fDirIndex * float4(proj_right * expandPixels / winSize.y, 0.0f, 0.0f) * ((vertexId & 0x01) ? hairEdgePositions[0].w : hairEdgePositions[1].w);
    Output.Tangent = float4(t, ratio);
    Output.p0p1 = float4(hairEdgePositions[0].xy / max(hairEdgePositions[0].w, TRESSFX_FLOAT_EPSILON), hairEdgePositions[1].xy / max(hairEdgePositions[1].w, TRESSFX_FLOAT_EPSILON));
    Output.strandColor = GetStrandColor(index);
    //Output.PosCheck = MatrixMult(g_mView, float4(v,1));

    return Output;

}

float3 ScreenToNDC(float3 vScreenPos, float4 viewport)
{
    float2 xy = vScreenPos.xy;

    // add viewport offset.
    xy += viewport.xy;

    // scale by viewport to put in 0 to 1
    xy /= viewport.zw;

    // shift and scale to put in -1 to 1. y is also being flipped.
    xy.x = (2 * xy.x) - 1;
    xy.y = 1 - (2 * xy.y);

    return float3(xy, vScreenPos.z);

}

float3 NDCToWorld(float3 vNDC, float4x4 mInvViewProj)
{
    float4 pos = mul(mInvViewProj, float4(vNDC, 1));

    return pos.xyz / pos.w;
}

// Allocate a new fragment location in fragment color, depth, and link buffers
int AllocateFragment(int2 vScreenAddress)
{
    int newAddress = LinkedListUAV.IncrementCounter();
    if (newAddress <= 0 || newAddress > nNodePoolSize)
        newAddress = FRAGMENT_LIST_NULL;
    return newAddress;
}

// Insert a new fragment at the head of the list. The old list head becomes the
// the second fragment in the list and so on. Return the address of the *old* head.
int MakeFragmentLink(int2 vScreenAddress, int nNewHeadAddress)
{
    int nOldHeadAddress;

    InterlockedExchange(tRWFragmentListHead[vScreenAddress], nNewHeadAddress, nOldHeadAddress);

    return nOldHeadAddress;
}

uint PackFloat4IntoUint(float4 vValue)
{
    return (((uint)(vValue.x * 255)) << 24) | (((uint)(vValue.y * 255)) << 16) | (((uint)(vValue.z * 255)) << 8) | (uint)(vValue.w * 255);
}

// Write fragment attributes to list location. 
void WriteFragmentAttributes(int nAddress, int nPreviousLink, float4 vData, float3 vColor3, float fDepth)
{
    PPLL_STRUCT element;
    element.data = PackFloat4IntoUint(vData);
    element.color = PackFloat4IntoUint(float4(vColor3, 0));
    element.depth = asuint(saturate(fDepth));
    element.uNext = nPreviousLink;
    LinkedListUAV[nAddress] = element;
}

#line 112 "oHair.sufx"


float4x4 g_mVP;
float3 g_vEye;
float4 g_vViewport;

struct PS_INPUT_HAIR
{
    float4 Position    : SV_POSITION;
    float4 Tangent     : Tangent;
    float4 p0p1        : TEXCOORD0;
    float3 strandColor : TEXCOORD1;
};


PS_INPUT_HAIR VS_RenderHair_AA(uint vertexId : SV_VertexID)
{
    TressFXVertex tressfxVert =
        GetExpandedTressFXVert(vertexId, g_vEye, g_vViewport.zw, g_mVP);

    PS_INPUT_HAIR Output;

    Output.Position = tressfxVert.Position;
    Output.Tangent = tressfxVert.Tangent;
    Output.p0p1 = tressfxVert.p0p1;
    Output.strandColor = tressfxVert.strandColor;

    return Output;
}

#if REAL_VOID
#	define VOID_RETURN void
#	define VOID_RETURN_SEMANTIC 
#	define RETURN_NOTHING return;
#	define RETURN_ERROR return;
#else
#	define VOID_RETURN float4
#	define VOID_RETURN_SEMANTIC : SV_TARGET
#	define RETURN_NOTHING return float4(0,0,0,0);
#	define RETURN_ERROR return float4(1,0,0,1);
#endif

#if ERROR_CHECK
#	define ASSERT(x) if(!(x)) RETURN_ERROR
#else
#	define ASSERT(x) 
#endif

#line 257 "oHair.sufx"

[earlydepthstencil]
VOID_RETURN ps_main(PS_INPUT_HAIR input) VOID_RETURN_SEMANTIC
{
    float3 vNDC = ScreenToNDC(input.Position.xyz, g_vViewport);
    float3 vPositionWS = NDCToWorld(vNDC, g_mInvViewProj);

    float coverage = ComputeCoverage(input.p0p1.xy, input.p0p1.zw, vNDC.xy, g_vViewport.zw);
    float alpha = coverage * g_MatBaseColor.a;

    ASSERT(coverage >= 0)
        if (alpha < 1.0 / 255.0)
            RETURN_NOTHING

            int2   vScreenAddress = int2(input.Position.xy);
            // Allocate a new fragment
            int nNewFragmentAddress = AllocateFragment(vScreenAddress);
            ASSERT(nNewFragmentAddress != FRAGMENT_LIST_NULL)

                int nOldFragmentAddress = MakeFragmentLink(vScreenAddress, nNewFragmentAddress);
            WriteFragmentAttributes(nNewFragmentAddress, nOldFragmentAddress, float4(input.Tangent.xyz * 0.5 + float3(0.5, 0.5, 0.5), alpha), input.strandColor.xyz, input.Position.z);

            RETURN_NOTHING


}

technique11 TressFX2
{
	pass P0
	{
        SetVertexShader(CompileShader(vs_5_0, VS_RenderHair_AA()));
        SetPixelShader(CompileShader(ps_5_0, ps_main()));
    }
}