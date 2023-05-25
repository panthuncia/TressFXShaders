

#line 146 "qHair.sufx"


float4x4 g_mInvViewProj;
float4 g_vViewport;
float3 g_vEye;


#line 6 "Effects/D3D11/iCommon.shl"


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

#line 152 "qHair.sufx"


#line 32 "Effects/D3D11/iCommon.shl"

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


// viewport.xy = offset, viewport.zw = size
float3 GetWorldPos(float4 vScreenPos, float4 viewport, float4x4 invViewProj)
{
    float2 xy = vScreenPos.xy;

    // add viewport offset.
    xy += viewport.xy;

    // scale by viewport to put in 0 to 1
    xy /= viewport.zw;

    // shift and scale to put in -1 to 1. y is also being flipped.
    xy.x = (2 * xy.x) + 1;
    xy.y = 1 - (2 * xy.y);

    float4 pos = mul(invViewProj, float4(xy.x, xy.y, vScreenPos.z, 1));
    //float4 pos = float4(xy.x, xy.y, 1, 1);
    //pos *= sv_pos.w;

    return pos.xyz / pos.w;
}

#line 153 "qHair.sufx"


#line 31 "../../amd_tressfx/src/shaders/TressFXRendering.hlsl"



#define AMD_PI 3.1415926
#define AMD_e 2.71828183

#define AMD_TRESSFX_KERNEL_SIZE 5

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







#line 154 "qHair.sufx"


#line 31 "../../amd_tressfx/src/shaders/TressFXPPLL.hlsl"

#if ERROR_CHECK
#	define ASSERT(x) if(!(x)) RETURN_ERROR
#else
#	define ASSERT(x) 
#endif


struct PPLL_STRUCT
{
    uint	depth;
    uint	data;
    uint    color;
    uint    uNext;
};

#ifndef FRAGMENT_LIST_NULL
#define FRAGMENT_LIST_NULL 0xffffffff
#endif

#define HAS_COLOR 1

#line 155 "qHair.sufx"




#line 33 "Effects/D3DCommon/Include/SuMath.shl"

#define SU_HALF_PI 1.57079632679
#define SU_PI 3.14159265359
#define SU_2PI 6.28318530718
#define SU_SQRT_PI 1.772453851
#define SU_SQRT2 1.41421356237
#define SU_EPSILON 1.e-7
#define SU_EPSILON_FLOAT 1.e-4
#define SU_MAX_FLOAT 1.e37

////////////////////////////////////////////////////////////////////////////
// Compute the [0,1] clamped dot3 for vec0 and vec1 
////////////////////////////////////////////////////////////////////////////
float SuDot3Clamp(float3 vec0, float3 vec1)
{
    return saturate(dot(vec0, vec1));
}

////////////////////////////////////////////////////////////////////////////
// SuPow(x,y)
// Returns pow( abs(x), y )
////////////////////////////////////////////////////////////////////////////

#ifdef SU_3D_API_D3D9
#define SuPow(x,y) pow(x,y)
#else
#define SuPow(x,y) pow( abs(x), y )
#endif

////////////////////////////////////////////////////////////////////////////
// SuSqrt(x)
//  Returns sqrt( abs(x) )
//  This function is provided to match the D3D9 sqrt() behavior
////////////////////////////////////////////////////////////////////////////
#ifdef SU_3D_API_D3D9
#define SuSqrt(x) sqrt(x)
#else
#define SuSqrt(x) sqrt( abs(x) )
#endif

////////////////////////////////////////////////////////////////////////////
// SuAtan2(x,y)
//   Atan2 function which duplicates D3D9's Atan2 behavior atan2(0,0) == 0
//    instead of atan2(0,0) == Nan
////////////////////////////////////////////////////////////////////////////
float SuAtan2(float x, float y)
{
    float ret = atan2(x, y);

#if defined(SU_3D_API_D3D10) || defined(SU_3D_API_D3D11)
    [flatten] if (x == 0 && y == 0)
    {
        return 0;
    }
#endif

    return ret;
}

////////////////////////////////////////////////////////////////////////////
// Convert degrees to radians
////////////////////////////////////////////////////////////////////////////
float SuDegreeToRad(float r)
{
    return r * (SU_PI / 180.0f);
}

////////////////////////////////////////////////////////////////////////////
// Convert radians to radians
////////////////////////////////////////////////////////////////////////////
float SuRadToDegree(float r)
{
    return r * (180.0f / SU_PI);
}

////////////////////////////////////////////////////////////////////////////
// Compute the reflection vector given a view vector and normal.
////////////////////////////////////////////////////////////////////////////
float3 SuReflect(float3 viewVec, float3 normal)
{
    return normalize(2 * dot(viewVec, normal) * normal - viewVec);
}

////////////////////////////////////////////////////////////////////////////
// Compute the reflection vector given a view vector, normal, and
// view dot normal
////////////////////////////////////////////////////////////////////////////
float3 SuReflect(float3 viewVec, float3 normal, float nDotV)
{
    return normalize((2 * nDotV * normal) - viewVec);
}

/////////////////////////////////////////////////////////////////////////////
// Computes an approximated fresnel term == [((1.0f - N.V)^5) * 0.95f] + 0.05f
/////////////////////////////////////////////////////////////////////////////
float SuComputeFresnelApprox(float3 normalVec, float3 viewVec)
{
    // (1-N.V)^4
    float NdotV5 = pow(1.0f - SuDot3Clamp(normalVec, viewVec), 4.0f);

    // scale and bias to fit to real fresnel curve
    return (NdotV5 * 0.95f) + 0.05f;
}

float SuComputeFresnelApprox(float NdotV)
{
    // (1-N.V)^4
    float NdotV5 = pow(1.0f - NdotV, 4.0f);

    // scale and bias to fit to real fresnel curve
    return (NdotV5 * 0.95f) + 0.05f;

}
/////////////////////////////////////////////////////////////////////////////
// Schlick's approximation of Fresnel
/////////////////////////////////////////////////////////////////////////////
float SchlickFresnel(float cosTheta, float R)
{
    return (R + SuPow(1.0 - cosTheta, 5.0) * (1.0 - R));
}

///////////////////////////////////////////////////////////////////////////////////
// Struck's Fresnel Term (good for approximating subsurface scattering in skin)
// I'm not sure of the original reference for this technique, see John Isidoro's 
// skin rendering smmary for discussion.
//
// fSoftnessMin and fSoftnessMax are constants controlling the range of luminance 
// values the fresnel term is applied to.
///////////////////////////////////////////////////////////////////////////////////
float SuStruckFresnel(float fLuminance, float fSoftnessMin, float fSoftnessMax, float3 vNormal, float3 vView)
{
    float fAttenuation = max(0, (fLuminance - fSoftnessMin)) / (fSoftnessMax - fSoftnessMin);
    float fNV = abs(dot(vNormal, vView));
    return (1 - fNV) * fAttenuation;
}

////////////////////////////////////////////////////////////////////////////
// Compute transmission direction T from incident direction I, normal N,
// going from medium with refractive index n1 to medium with refractive
// index n2, with refraction governed by Snell's law:
//               n1*sin(theta1) = n2*sin(theta2).
// If there is total internal reflection, return 0, else set T and
// return 1. All vectors unit.
// From: http://research.microsoft.com/~hollasch/cgindex/render/refraction.txt
////////////////////////////////////////////////////////////////////////////
float3 SuTransmissionDirection(float fromIR, float toIR,
    float3 incoming, float3 normal)
{
    float eta = fromIR / toIR; // relative index of refraction
    float c1 = -dot(incoming, normal); // cos(theta1)
    float cs2 = 1. - eta * eta * (1. - c1 * c1); // cos^2(theta2)
    float3 v = (eta * incoming + (eta * c1 - sqrt(cs2)) * normal);
    if (cs2 < 0.) v = 0; // total internal reflection
    return v;
}

////////////////////////////////////////////////////////////////////////////
// Compute the full Fresnel terms going from one index of refraction to
// another. Returns both the transmitted term and the reflected terms.
////////////////////////////////////////////////////////////////////////////
float3 SuFullFresnel(float fromIR, float toIR, float3 incoming,
    float3 normal, out float reflect, out float transmit)
{
    float3 Tvec = SuTransmissionDirection(fromIR, toIR, incoming, normal);
    float NdotOmega = dot(normal, incoming);
    float NdotT = dot(normal, Tvec);
    float rPar = (toIR * NdotOmega + fromIR * NdotT) /
        (toIR * NdotOmega - fromIR * NdotT);
    float rPerp = (fromIR * NdotOmega + toIR * NdotT) /
        (fromIR * NdotOmega - toIR * NdotT);
    reflect = 0.5 * (rPar * rPar + rPerp * rPerp);
    float eta = fromIR / toIR; // relative index of refraction
    transmit = (1.0 - reflect) * (eta * eta);
    return Tvec;
}

////////////////////////////////////////////////////////////////////////////
// Compute the diffuse falloff hack defined by Mr. Alex Vlachos.  This 
// causes N.L to prematurely falloff, thus masking the hard edge that would 
// otherwise be visible.
////////////////////////////////////////////////////////////////////////////
float SuComputeDiffuseHack(float diffuseNdotL)
{
    return saturate((diffuseNdotL * (5.0f / 4.0f)) - (1.0f / 4.0f));
}

////////////////////////////////////////////////////////////////////////////
// Compute the diffuse falloff hack defined by Mr. Alex Vlachos.  This 
// causes N.L to prematurely falloff, thus masking the hard edge that would 
// otherwise be visible.
// diffuseBumpNdotL is N.L with N coming from bump map
// diffuseNdotL is N.L with N coming from geometry (lightVec.z in tangent space)
////////////////////////////////////////////////////////////////////////////
float SuComputeDiffuseBumpHack(float diffuseNdotL, float diffuseBumpNdotL)
{
    float hack = SuComputeDiffuseHack(diffuseBumpNdotL);
    hack *= 1.0f - (pow(1.0f - SuComputeDiffuseHack(diffuseNdotL), 8.0f));

    return saturate(hack);
}

////////////////////////////////////////////////////////////////////////////
// Computes R.L^k where k = (exponent * (max - min)) + min
// This assumes reflectionVec and lightVec are normalized vectors
////////////////////////////////////////////////////////////////////////////
float SuComputeSpecular(float3 reflectionVec, float3 lightVec, float exponent, float min, float max)
{
    float scale = max - min;
    float k = (exponent * scale) + min;
    float RL = SuDot3Clamp(reflectionVec, lightVec);

    return pow(RL, k);
}

////////////////////////////////////////////////////////////////////////////
// Turn a color into a grayscale value.
////////////////////////////////////////////////////////////////////////////
float SuMakeWeightedGrayScale(float3 color)
{
    return dot(color, float3(0.30f, 0.59f, 0.11f));
}

float SuMakeGrayScale(float3 color)
{
    return dot(color, float3(0.333f, 0.333f, 0.333f));
}

////////////////////////////////////////////////////////////////////////////
// Pack a vector into a color 
////////////////////////////////////////////////////////////////////////////
float4 SuConvertVectorToColor(float4 vec)
{
    return ((vec * 0.5f) + 0.5f);
}

float3 SuConvertVectorToColor(float3 vec)
{
    return ((vec * 0.5f) + 0.5f);
}

////////////////////////////////////////////////////////////////////////////
// Convert from RGB color space into YUV color space
////////////////////////////////////////////////////////////////////////////
float3 SuRGBToYUV(float3 color)
{
    // http://en.wikipedia.org/wiki/YUV
    float Y = dot(color.xyz, float3(0.299, 0.587, 0.114)); // Luminance
    float U = 0.436 * (color.b - Y) / (1 - 0.114);
    float V = 0.615 * (color.r - Y) / (1 - 0.299);

    return float3(Y, U, V);
}

////////////////////////////////////////////////////////////////////////////
// Convert from YUV color space into RGB color space
////////////////////////////////////////////////////////////////////////////
float3 SuYUVToRGB(float3 color)
{
    return mul(float3x3(1, 0, 1.3983,
        1, -0.39465, -0.58060,
        1, 2.03211, 0), color);
}

////////////////////////////////////////////////////////////////////////////
// Convert HDR floating point RGB values into RGBE 
////////////////////////////////////////////////////////////////////////////
float4 SuEncodeRGBE(float3 rgb)
{
    float4 vEncoded;

    // Determine the largest color component
    float maxComponent = max(max(rgb.r, rgb.g), rgb.b);

    // Round to the nearest integer exponent
    float fExp = ceil(log2(maxComponent));

    // Divide the components by the shared exponent
    vEncoded.rgb = rgb / exp2(fExp);

    // Store the shared exponent in the alpha channel
    vEncoded.a = (fExp + 128) / 255;

    return vEncoded;
}

////////////////////////////////////////////////////////////////////////////
// Convert RGBE8 texels into HDR floating point RGB values
////////////////////////////////////////////////////////////////////////////
float3 SuDecodeRGBE(float4 rgbe)
{
    float3 vDecoded;

    // Retrieve the shared exponent
    float fExp = rgbe.a * 255 - 128;

    // Multiply through the color components
    vDecoded = rgbe.rgb * exp2(fExp);

    return vDecoded;
}

////////////////////////////////////////////////////////////////////////////
// Bilinear sample an RGBE8 texture
////////////////////////////////////////////////////////////////////////////
float3 SuBilinearFilterRGBE(sampler tLightMap, float2 vTexCoord, float2 vTexelSize, float2 vTextureSize)
{
    static const float4  mTexelOffsets[4] = { { 0.0,  0.0, 0, 0},
                                              { 1.0,  0.0, 0, 0},
                                              { 0.0,  1.0, 0, 0},
                                              { 1.0,  1.0, 0, 0} };

    float2 vTexCoordsTexel = vTexCoord * vTextureSize;

    //find min of change in u and v across quad
    float2 vDx = ddx(vTexCoordsTexel);
    float2 vDy = ddy(vTexCoordsTexel);

    float2 vDCoords;
    vDCoords = vDx * vDx;
    vDCoords += vDy * vDy;  //compute du and dv magnitude across quad

    //standard mipmapping uses max here
    float fMaxTexCoordDelta = max(vDCoords.x, vDCoords.y);

    //compute mip level  (* 0.5 is effectively computing a square root before the )
    float fMipLevel = 0.5 * log2(fMaxTexCoordDelta);
    fMipLevel = max(fMipLevel, 0);

    float fMipLevelInt;
    float fMipLevelFrac;

    fMipLevelFrac = modf(fMipLevel, fMipLevelInt);

    float4 vNewCoord = float4(vTexCoord.xy, fMipLevelInt, fMipLevelInt);

    //texelOffset in texture coordinates for this miplevel	
    //float4 vFullTexelOffset = float4( vTexelSize * pow(2,fMipLevelInt) , 0.0, 0.0 );
    float4 vFullTexelOffset = float4(ldexp(vTexelSize, fMipLevelInt), 0.0, 0.0);

    // Texture size for this miplevel
     //mipLevelSize = g_vShadowMapSize / (2^mipLevelInt);
    float2 vMipLevelSize = ldexp(vTextureSize, -fMipLevelInt);

    vNewCoord.xy -= (1.0 / vMipLevelSize) * 0.5;


    // Fractional component of texel offset for bilinear blending
    float2 vFrac = frac(vNewCoord.xy * vMipLevelSize.xy);

    //return SuDecodeRGBE( tex2D( tLightMap, vTexCoord ) );


    //sample nearest 2x2 quad
    float3 cColor0 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord));
    float3 cColor1 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[1] * vFullTexelOffset)));
    float3 cColor2 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[2] * vFullTexelOffset)));
    float3 cColor3 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[3] * vFullTexelOffset)));

    //bilinear interpolation weights
    float4 vBilinearWeights = float4(1.0 - vFrac.x, vFrac.x, 1.0 - vFrac.x, vFrac.x);
    vBilinearWeights *= float4(1.0 - vFrac.y, 1.0 - vFrac.y, vFrac.y, vFrac.y);

    return (cColor0 * vBilinearWeights.x + cColor1 * vBilinearWeights.y + cColor2 * vBilinearWeights.z + cColor3 * vBilinearWeights.w);
}

////////////////////////////////////////////////////////////////////////////
// Trilinear sample an RGBE8 texture
////////////////////////////////////////////////////////////////////////////
float3 SuTrilinearFilterRGBE(sampler tLightMap, float2 vTexCoord, float2 vTexelSize, float2 vTextureSize)
{

    static const float4  mTexelOffsets[4] = { { 0.0,  0.0, 0, 0},
                                              { 1.0,  0.0, 0, 0},
                                              { 0.0,  1.0, 0, 0},
                                              { 1.0,  1.0, 0, 0} };

    float2 vTexCoordsTexel = vTexCoord * vTextureSize;

    //find min of change in u and v across quad
    float2 vDx = ddx(vTexCoordsTexel);
    float2 vDy = ddy(vTexCoordsTexel);

    float2 vCoords;
    vCoords = vDx * vDx;
    vCoords += vDy * vDy;  //compute du and dv magnitude across quad

    //standard mipmapping uses max here
    float fMaxTexCoordDelta = max(vCoords.x, vCoords.y);

    //compute mip level  (* 0.5 is effectively computing a square root before the )
    float fMipLevel = 0.5 * log2(fMaxTexCoordDelta);
    fMipLevel = max(fMipLevel, 0);

    float fMipLevelInt;
    float fMipLevelFrac;

    fMipLevelFrac = modf(fMipLevel, fMipLevelInt);

    float4 vNewCoord = float4(vTexCoord.xy, fMipLevelInt, fMipLevelInt);

    //texelOffset in texture coordinates for this miplevel	
     //fullTexelOffset = g_vFullTexelOffset * (2^mipLevelInt);
    float4 vFullTexelOffset = float4(ldexp(vTexelSize, fMipLevelInt), 0.0, 0.0);

    // Texture size for this miplevel
     //mipLevelSize = g_vShadowMapSize / (2^mipLevelInt);
    float2 vMipLevelSize = ldexp(vTextureSize, -fMipLevelInt);
    vNewCoord.xy -= (1.0 / vMipLevelSize) * 0.5;

    //bias for this miplevel (could possibly be pre-evaluated in the shadow map)
     // double bias each layer down the mipchain
     //mipLevelBias = ldexp( g_fDistBias, fMipLevelInt );

    // Fractional component of texel offset for bilinear blending
    float2 vFrac = frac(vNewCoord.xy * vMipLevelSize.xy);

    //sample nearest 2x2 quad
    float3 cColor0 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord));
    float3 cColor1 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[1] * vFullTexelOffset)));
    float3 cColor2 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[2] * vFullTexelOffset)));
    float3 cColor3 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[3] * vFullTexelOffset)));

    //bilinear interpolation weights
    float4 vBilinearWeights = float4(1.0 - vFrac.x, vFrac.x, 1.0 - vFrac.x, vFrac.x);
    vBilinearWeights *= float4(1.0 - vFrac.y, 1.0 - vFrac.y, vFrac.y, vFrac.y);

    float3 cSample = (1.0 - fMipLevelFrac) * (cColor0 * vBilinearWeights.x + cColor1 * vBilinearWeights.y + cColor2 * vBilinearWeights.z + cColor3 * vBilinearWeights.w);

    // set next mip level
    vNewCoord.zw += 1.0;

    //texelOffset in texture coordinates for this miplevel	
    vFullTexelOffset = vFullTexelOffset * 2.0f;

    // Texture size for this miplevel
    vMipLevelSize = vMipLevelSize * 0.5f;
    vNewCoord.xy -= (1.0 / vMipLevelSize) * 0.25;

    // Fractional component of texel offset for bilinear blending
    vFrac = frac(vNewCoord.xy * vMipLevelSize.xy);

    //bilinear interpolation weights
    vBilinearWeights = float4(1.0 - vFrac.x, vFrac.x, 1.0 - vFrac.x, vFrac.x);
    vBilinearWeights *= float4(1.0 - vFrac.y, 1.0 - vFrac.y, vFrac.y, vFrac.y);

    //sample nearest 2x2 quad
    cColor0 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord));
    cColor1 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[1] * vFullTexelOffset)));
    cColor2 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[2] * vFullTexelOffset)));
    cColor3 = SuDecodeRGBE(tex2Dlod(tLightMap, vNewCoord + (mTexelOffsets[3] * vFullTexelOffset)));

    cSample += fMipLevelFrac * (cColor0 * vBilinearWeights.x + cColor1 * vBilinearWeights.y + cColor2 * vBilinearWeights.z + cColor3 * vBilinearWeights.w);

    return cSample;
}


////////////////////////////////////////////////////////////////////////////
// Distributes an integer range across four color channels [0, MAXINT] -> <[0,1], [0,1], [0,1]>
////////////////////////////////////////////////////////////////////////////
float3 SuEncodeUIntAsColor(int i)
{

    float3 cOut = 0;
    i = max(i, 0); // this won't encode signed values

    cOut.r = saturate((i % 256) / 255.0f);
    cOut.g = saturate(((i / 256) % 256) / 255.0f);
    cOut.b = saturate(((i / 65536) % 256) / 255.0f);

    return cOut;
}

int SuDecodeUIntAsColor(float3 color)
{
    return dot(color, float3(255.0f, 65535.0f, 16777215.0f));
}


////////////////////////////////////////////////////////////////////////////
// Un Pack a vector from a color 
////////////////////////////////////////////////////////////////////////////
float4 SuConvertColorToVector(float4 color)
{
    return ((color * 2.0f) - 1.0f);
}

float3 SuConvertColorToVector(float3 color)
{
    return ((color * 2.0f) - 1.0f);
}

////////////////////////////////////////////////////////////////////////////
// Get the shadow term for a given number of lights. The number of lights
// term determines how close to the "true" ambient term to return. For
// a single light you only get the ambient term, for two lights you get
// half the real light, three lights you get two thirds of the full light.
////////////////////////////////////////////////////////////////////////////
float SuComputeShadowDimFactor(float3 ambientLight, float3 combinedLight,
    int numLights)
{
    return lerp((SuMakeGrayScale(ambientLight) / SuMakeGrayScale(combinedLight)), 1.0, (((float)numLights - 1.0f) / numLights));
}

////////////////////////////////////////////////////////////////////////////
// As with the above function this one returns the ambient term, but is 
// more optimized since we know there is no lerp involved.
////////////////////////////////////////////////////////////////////////////
float SuComputeShadowDimFactorOneLight(float3 ambientLight,
    float3 combinedLight)
{
    return (SuMakeGrayScale(ambientLight) / SuMakeGrayScale(combinedLight));
}

float SuGetLuminance(float3 color)
{
    return (dot(color, float3(0.3, 0.59, 0.11)));
}

////////////////////////////////////////////////////////////////////////////
// Compute luminance value from a given RGB color using formula from
// Reinhard paper (http://www.cs.ucf.edu/~reinhard/papers/jgt_reinhard.pdf)
////////////////////////////////////////////////////////////////////////////
float SuGetReinhardLuminance(float3 cRGBValue)
{
    return dot(cRGBValue, float3(0.27, 0.67, 0.06));
}

////////////////////////////////////////////////////////////////////////////
// Compute Z and scale ATI2N bump maps
////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////
// Compute Z and scale ATI2N bump maps
////////////////////////////////////////////////////////////////////////////
float3 SuComputeNormalATI2Nxy(float2 v, float xyscale)
{
    v = v * 2.0 - 1.0;
#ifdef SU_3D_API_D3D9
    // We used to use a dot product here, but the D3D9 Dec. SDK HLSL compiler screws it all up...
    float3 r = float3(v, SuSqrt(1.0 - v.x * v.x - v.y * v.y));
#else
    float3 r = float3(v, SuSqrt(1.0 - dot(v, v)));
#endif
    r.xy *= xyscale;
    return normalize(r);
}

float3 SuComputeNormalATI2N(float2 v, float zscale)
{
    v = v * 2.0 - 1.0;
#ifdef SU_3D_API_D3D9
    // We used to use a dot product here, but the D3D9 Dec. SDK HLSL compiler screws it all up...
    float3 r = float3(v, SuSqrt(1.0 - v.x * v.x - v.y * v.y));
#else
    float3 r = float3(v, SuSqrt(1.0 - dot(v, v)));
#endif
    r.z *= zscale;
    return normalize(r);
}

float3 SuComputeNormalATI2N(float2 v)
{
    v = v * 2.0 - float2(1.0, 1.0);
#ifdef SU_3D_API_D3D9
    // We used to use a dot product here, but the D3D9 Dec. SDK HLSL compiler screws it all up...
    float3 r = float3(v, SuSqrt(1.0 - v.x * v.x - v.y * v.y));
#else
    float3 r = float3(v, SuSqrt(1.0 - dot(v, v)));
#endif
    return r;
}

// Decodes a dec3n vertex shader input element and brings it into [-1, 1] range.
float3 SuDecodeDec3N(float3 vVal)
{
#ifdef SU_3D_API_D3D9
    return vVal;      // D3D9
#else
    return 2.0 * vVal - 1.0;   // D3D10 & D3D11
#endif
}

//==========================================================//
// Convert cartesian coordinates into spherical coordinates //
// returns <theta, phi>                                     //
//==========================================================//
float2 SuCartesian2Spherical(float3 cartesian)
{
    float theta;
    float phi;

    theta = acos(cartesian.z);

    float s = cartesian.x * cartesian.x + cartesian.y * cartesian.y;
    if (s <= 0.0)
    {
        phi = 0.0;
    }
    else
    {
        s = sqrt(s);

        if (cartesian.x >= 0.0)
        {
            phi = asin(cartesian.y / s);
        }
        else
        {
            phi = 3.14159265359 - asin(cartesian.y / s);
        }
    }

    return float2(theta, phi);
}

////////////////////////////////////////////////////////////////////////////
// Decode the Sushi packing of 1010102 "HDR"
////////////////////////////////////////////////////////////////////////////
float3 SuDecode1010102HDR(float4 hdrSample)
{
    return hdrSample.rgb * 4.0 * (hdrSample.a * 3.0 + 1.0);
}

////////////////////////////////////////////////////////////////////////////
// Decode the Sushi packing of 1010102 "HDR"
////////////////////////////////////////////////////////////////////////////
float3 SuDecode1010102HDRDeGamma(float4 hdrSample)
{
    return pow(hdrSample.rgb * 4.0 * (hdrSample.a * 3.0 + 1.0), 1 / 2.2);
}

// Similar interface to smoothstep, but does a straight linear function
// between min and max
float4 SuLerpStep(float minV, float maxV, float4 value)
{
    return saturate((value - minV) / (maxV - minV));
}

// Similar interface to smoothstep, but does a straight linear function
// between min and max
float3 SuLerpStep(float minV, float maxV, float3 value)
{
    return saturate((value - minV) / (maxV - minV));
}

// Similar interface to smoothstep, but does a straight linear function
// between min and max
float2 SuLerpStep(float minV, float maxV, float2 value)
{
    return saturate((value - minV) / (maxV - minV));
}

// Similar interface to smoothstep, but does a straight linear function
// between min and max
float SuLerpStep(float minV, float maxV, float value)
{
    return saturate((value - minV) / (maxV - minV));
}

////////////////////////////////////////////////////////////////////////////
// Computes the area of intersection of two spherical caps.
// r0 : radius of spherical cap 0 (radians)
// r1 : radius of spherical cap 1 (radians)
// d  : distance between cetroids of cap 0 and cap 1 (radians)
////////////////////////////////////////////////////////////////////////////
float SuSphericalCapIntersectionArea(float r0, float r1, float d)
{
    float fArea;

    if (min(r0, r1) <= max(r0, r1) - d)
    {
        // One cap in completely inside the other
        fArea = 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }
    else if (r0 + r1 <= d)
    {
        // No intersection exists
        fArea = 0;
    }
    else
    {
        fArea = 2.0 * acos((-cos(d) + cos(r0) * cos(r1)) / (sin(r0) * sin(r1))) -
            2.0 * acos((cos(r1) - cos(d) * cos(r0)) / (sin(d) * sin(r0))) * cos(r0) -
            SU_2PI * cos(r1) +
            2.0 * cos(r1) * acos((-cos(r0) + cos(d) * cos(r1)) / (sin(d) * sin(r1)));
        /*
         fArea = 2.0 * ( -acos( cos(d) * csc(r0) * csc(r1) - cot(r0) * cot(r1) ) -
                  acos( cos(r1) * csc(d) * csc(r0) - cot(d) * cot(r0) ) * cos(r0) -
                  acos( cos(r0) * csc(d) * csc(r1) - cot(d) * cot(r1) ) * cos(r1) + SU_PI);
        */
    }

    return fArea;
}

float SuSphericalCapIntersectionAreaFast(float r0, float r1, float d)
{
    float fArea;

    if (d <= max(r0, r1) - min(r0, r1))
    {
        // One cap in completely inside the other
        fArea = 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }
    else if (d >= r0 + r1)
    {
        // No intersection exists
        fArea = 0;
    }
    else
    {
        float b = abs(r0 - r1);
        fArea = (1.0 - saturate((d - b) / (r0 + r1 - b))) * (6.283185308 - 6.283185308 * cos(min(r0, r1)));
    }

    return fArea;
}

float SuSphericalCapIntersectionAreaFastNoDyn(float r0, float r1, float d)
{
    float b = abs(r0 - r1);
    return (1.0 - saturate((d - b) / (r0 + r1 - b))) * (6.283185308 - 6.283185308 * cos(min(r0, r1)));
}

float SuSphericalCapIntersectionAreaFast2(float r0, float r1, float d)
{
    float fArea;

    if (d <= abs(r0 - r1))
    {
        // One cap in completely inside the other
        fArea = 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }
    else if (d >= r0 + r1)
    {
        // No intersection exists
        fArea = 0;
    }
    else
    {
        float b = abs(r0 - r1);
        fArea = smoothstep(0.0, 1.0, 1.0 - saturate((d - b) / (r0 + r1 - b)));
        fArea *= 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }

    return fArea;
}

float SuSphericalCapIntersectionAreaFast3(float r0, float r1, float d)
{
    float fArea;

    if (d <= abs(r0 - r1))
    {
        // One cap in completely inside the other
        fArea = 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }
    else if (d >= r0 + r1)
    {
        // No intersection exists
        fArea = 0;
    }
    else
    {
        float b = abs(r0 - r1);
        fArea = SuLerpStep(0.0, 1.0, 1.0 - saturate((d - b) / (r0 + r1 - b)));
        fArea *= 6.283185308 - 6.283185308 * cos(min(r0, r1));
    }

    return fArea;
}

float SuSphericalCapIntersectionAreaFast2NoDyn(float r0, float r1, float d)
{
    float b = abs(r0 - r1);
    return smoothstep(0.0, 1.0, 1.0 - ((d - b) / (r0 + r1 - b))) * (6.283185308 - 6.283185308 * cos(min(r0, r1)));
}

/////////////////////////////////////////////////////////////////////////////
// Note: this assumes sphere space coordinates: the center of the sphere is at (0,0,0)
// fSphereRadiusSquared = radius squared
// vRayOrigin = ray original (point)
// vRayDirection = ray direction (vector)
/////////////////////////////////////////////////////////////////////////////

float SuSphereIntersect(float fSphereRadiusSquared, float3 vRayOrigin, float3 vRayDirection)
{
    float3 v = float3(0, 0, 0) - vRayOrigin;
    float b = dot(v, vRayDirection);
    float disc = (b * b) - dot(v, v) + fSphereRadiusSquared;
    disc = SuSqrt(disc);
    float t2 = b + disc;
    float t1 = b - disc;
    return max(t1, t2);
}

/////////////////////////////////////////////////////////////////////////////
// Converts a Z-buffer value to a camera-space distance
/////////////////////////////////////////////////////////////////////////////
float SuNDCDepthToCameraDepth(float fNDCDepth, float4 vCamParams)
{
    return (-vCamParams.z * vCamParams.w) / (fNDCDepth * (vCamParams.w - vCamParams.z) - vCamParams.w);
}


/////////////////////////////////////////////////////////////////////////////
float SuBarycentricLerp(float f0, float f1, float f2, float3 vBarycentrics)
{
    return f0 * vBarycentrics.x + f1 * vBarycentrics.y + f2 * vBarycentrics.z;
}

/////////////////////////////////////////////////////////////////////////////
float2 SuBarycentricLerp(float2 f0, float2 f1, float2 f2, float3 vBarycentrics)
{
    return f0.xy * vBarycentrics.x + f1.xy * vBarycentrics.y + f2.xy * vBarycentrics.z;
}

/////////////////////////////////////////////////////////////////////////////   
float3 SuBarycentricLerp(float3 f0, float3 f1, float3 f2, float3 vBarycentrics)
{
    return f0.xyz * vBarycentrics.x + f1.xyz * vBarycentrics.y + f2.xyz * vBarycentrics.z;
}

/////////////////////////////////////////////////////////////////////////////
float4 SuBarycentricLerp(float4 f0, float4 f1, float4 f2, float3 vBarycentrics)
{
    return f0.xyzw * vBarycentrics.x + f1.xyzw * vBarycentrics.y + f2.xyzw * vBarycentrics.z;
}

#line 158 "qHair.sufx"


#line 34 "Effects/D3DCommon/Include/SuGamma.shl"

float SuLinearToGamma(float fVal) { return pow(fVal, SU_LINEAR_TO_sRGB); }
float SuGammaToLinear(float fVal) { return pow(fVal, SU_sRGB_TO_LINEAR); }

float2 SuLinearToGamma(float2 vVal) { return pow(vVal, SU_LINEAR_TO_sRGB); }
float2 SuGammaToLinear(float2 vVal) { return pow(vVal, SU_sRGB_TO_LINEAR); }

float3 SuLinearToGamma(float3 vVal) { return pow(vVal, SU_LINEAR_TO_sRGB); }
float3 SuGammaToLinear(float3 vVal) { return pow(vVal, SU_sRGB_TO_LINEAR); }

float4 SuLinearToGamma(float4 vVal) { return pow(vVal, SU_LINEAR_TO_sRGB); }
float4 SuGammaToLinear(float4 vVal) { return pow(vVal, SU_sRGB_TO_LINEAR); }

#line 159 "qHair.sufx"


#line 78 "Effects/D3D11/iShadows.shl"

Texture2DArray<float> tShadowMap;
SamplerComparisonState sComparison;
SamplerComparisonState sComparisonPCF;

sampler sPointBorder;

StructuredBuffer<float4x4> ShadowMatrices;
StructuredBuffer<float4> ShadowParams;

float4x4  mShadowVPT;
float4 vShadowMapDimensions;

bool bEnableShadow;

#define BRIGHT_OUTSIDE_FRUSTUM 1

bool FrustumCheck(float3 vPositionSM)
{
#if BRIGHT_OUTSIDE_FRUSTUM
    if (max(vPositionSM.x, vPositionSM.y) > 1.) { return false; }
    if (min(vPositionSM.x, vPositionSM.y) < 0.) { return false; }
    if (vPositionSM.z > 1.) return false;
    if (vPositionSM.z < 0.) return false;
#endif
    return true;
}

//=================================================================
//=================================================================
float3 ComputeShadowUV(float3 vPositionWS, int nSMIndex = 0)
{
    // Project position into shadow map
    //float4 vPositionSM = mul( mShadowVPT, float4( vPositionWS, 1 ) );
    float4 vPositionSM = mul(ShadowMatrices[nSMIndex], float4(vPositionWS, 1));
    return vPositionSM.xyz / vPositionSM.www;
}


//=================================================================
// Single tap comparison
//=================================================================
float ComputeShadow(float3 vPositionWS, float nSMIndex = 0)
{
    float3 vPositionSM = ComputeShadowUV(vPositionWS, nSMIndex);

    // if ( max(vPositionSM.x,vPositionSM.y) > 1. ) { return 0; }
     //if ( min(vPositionSM.x,vPositionSM.y) < 0. ) { return 0; }

    float fShadow = tShadowMap.SampleCmpLevelZero(sComparison, float3(vPositionSM.xy, nSMIndex), vPositionSM.z - SM_CONST_BIAS);
    return fShadow;
}

//=================================================================
// Single tap comparison
//=================================================================
float ComputeShadow(float2 vUV, float fDepth, float nSMIndex = 0)
{
    float fShadow = tShadowMap.SampleCmpLevelZero(sComparison, float3(vUV, nSMIndex), fDepth - SM_CONST_BIAS);
    return fShadow;
}

//=================================================================
// 2x2 PCF. 5 shades of gray.
//=================================================================
float ComputeShadowPCF(float3 vPositionWS, float nSMIndex = 0)
{
    float3 vPositionSM = ComputeShadowUV(vPositionWS, nSMIndex);

    if (!FrustumCheck(vPositionSM)) return 1.0;

    float fShadow = tShadowMap.SampleCmpLevelZero(sComparisonPCF, float3(vPositionSM.xy, nSMIndex), vPositionSM.z - SM_CONST_BIAS);
    return fShadow;
}

//=================================================================
// Shadow map sampling with 28 tap poisson kernel (each Poisson tap 
// is a 2x2 PCF kernel). This gives 140 shades of gray.
//=================================================================
float ComputeShadowPoisson(float3 vPositionWS, float fRadius = 5.0, float nSMIndex = 0)
{
    const float2 vPoisson[27] = { float2(-0.525820f, -0.127704f),
                                  float2(0.595566f,  0.784995f),
                                  float2(-0.374618f, -0.460896f),
                                  float2(0.646400f,  0.436244f),
                                  float2(-0.001001f,  0.271255f),
                                  float2(0.943513f, -0.289188f),
                                  float2(-0.272002f,  0.515921f),
                                  float2(-0.952234f, -0.078234f),
                                  float2(-0.758021f,  0.217861f),
                                  float2(0.073475f, -0.554726f),
                                  float2(-0.621979f, -0.768835f),
                                  float2(0.268312f,  0.538478f),
                                  float2(0.412263f,  0.171512f),
                                  float2(-0.148248f,  0.979633f),
                                  float2(-0.726008f,  0.630549f),
                                  float2(0.212817f, -0.188554f),
                                  float2(-0.279090f, -0.893269f),
                                  float2(0.114498f, -0.973203f),
                                  float2(0.518764f, -0.453969f),
                                  float2(0.728637f, -0.027399f),
                                  float2(-0.164580f, -0.109996f),
                                  float2(0.206435f,  0.970726f),
                                  float2(-0.779092f, -0.445420f),
                                  float2(0.416586f, -0.806773f),
                                  float2(0.950401f,  0.277201f),
                                  float2(-0.341163f,  0.182236f),
                                  float2(-0.471442f,  0.867417f) };

    float3 vPositionSM = ComputeShadowUV(vPositionWS, nSMIndex);
    if (!FrustumCheck(vPositionSM)) return 1.0;

    float2 vTexelSize = 1.0 / vShadowMapDimensions.xy;
    float fShadow = tShadowMap.SampleCmpLevelZero(sComparisonPCF, float3(vPositionSM.xy, nSMIndex), vPositionSM.z - SM_CONST_BIAS);

    [unroll] for (int nIndex = 0; nIndex < 27; nIndex++)
    {
        fShadow += tShadowMap.SampleCmpLevelZero(sComparisonPCF, float3(vPositionSM.xy + (vTexelSize * vPoisson[nIndex] * fRadius.xx), nSMIndex), vPositionSM.z - SM_CONST_BIAS);
    }

    return fShadow / 28.0;
}

//=================================================================
// Shadow map sampling with 28 tap poisson kernel (each Poisson tap 
// is a 2x2 PCF kernel). This gives 140 shades of gray.
//=================================================================
float ComputeShadowPoisson10(float3 vPositionWS, float fRadius = 5.0, float nSMIndex = 0)
{
    const float2 vPoisson[10] = {
        float2((0.079928 - 0.5) * 2,       (0.995178 - 0.5) * 2),
        float2((0.120334 - 0.5) * 2,       (0.710807 - 0.5) * 2),
        float2((0.186102 - 0.5) * 2,       (0.400647 - 0.5) * 2),
        float2((0.386639 - 0.5) * 2,       (0.992065 - 0.5) * 2),
        float2((0.448134 - 0.5) * 2,       (0.469771 - 0.5) * 2),
        float2((0.620014 - 0.5) * 2,       (0.666311 - 0.5) * 2),
        float2((0.719260 - 0.5) * 2,       (0.385296 - 0.5) * 2),
        float2((0.720695 - 0.5) * 2,       (0.990722 - 0.5) * 2),
        float2((0.888516 - 0.5) * 2,       (0.187658 - 0.5) * 2),
        float2((0.890286 - 0.5) * 2,       (0.591052 - 0.5) * 2) };

    float3 vPositionSM = ComputeShadowUV(vPositionWS, nSMIndex);
    if (!FrustumCheck(vPositionSM)) return 1.0;

    float2 vTexelSize = 1.0 / vShadowMapDimensions.xy;
    float fShadow = tShadowMap.SampleCmpLevelZero(sComparisonPCF, float3(vPositionSM.xy, nSMIndex), vPositionSM.z - SM_CONST_BIAS);
    //float fShadow = 0;

    [unroll] for (int nIndex = 0; nIndex < 10; nIndex++)
    {
        fShadow += tShadowMap.SampleCmpLevelZero(sComparisonPCF, float3(vPositionSM.xy + (vTexelSize * vPoisson[nIndex] * fRadius.xx), nSMIndex), vPositionSM.z - SM_CONST_BIAS);
    }

    return fShadow / 11.0;
}



float GetWSDepth_D3D(float depthNDC, float fNear, float fFar)
{
    return fNear * fFar / (fFar - depthNDC * (fFar - fNear));
}


//=================================================================
// Shadow map sampling with 28 tap poisson kernel (each Poisson tap 
// is a 2x2 PCF kernel). This gives 140 shades of gray.
//=================================================================
float ComputeShadowHair(float3 vPositionWS, float fRadius, float nSMIndex)
{
    const float2 vPoisson[11] = {
        float2(0, 0),
        float2((0.079928 - 0.5) * 2,       (0.995178 - 0.5) * 2),
        float2((0.120334 - 0.5) * 2,       (0.710807 - 0.5) * 2),
        float2((0.186102 - 0.5) * 2,       (0.400647 - 0.5) * 2),
        float2((0.386639 - 0.5) * 2,       (0.992065 - 0.5) * 2),
        float2((0.448134 - 0.5) * 2,       (0.469771 - 0.5) * 2),
        float2((0.620014 - 0.5) * 2,       (0.666311 - 0.5) * 2),
        float2((0.719260 - 0.5) * 2,       (0.385296 - 0.5) * 2),
        float2((0.720695 - 0.5) * 2,       (0.990722 - 0.5) * 2),
        float2((0.888516 - 0.5) * 2,       (0.187658 - 0.5) * 2),
        float2((0.890286 - 0.5) * 2,       (0.591052 - 0.5) * 2) };

    float3 vPositionSM = ComputeShadowUV(vPositionWS, nSMIndex);
    if (!FrustumCheck(vPositionSM)) return 1.0;

    float2 vTexelSize = 1.0 / vShadowMapDimensions.xy;

    float fNear = ShadowParams[int(nSMIndex)].z;
    float fFar = ShadowParams[int(nSMIndex)].w;
    float fDepthFragment = GetWSDepth_D3D(vPositionSM.z, fNear, fFar);


    float fShadow = 0;

    float fTotalWeight = 0;

    for (int iSample = 0; iSample < 11; ++iSample)
    {

        float fWeight = 1.0;

        float fDepthSM_NDC = tShadowMap.SampleLevel(sPointBorder, float3(vPositionSM.xy + (vTexelSize * vPoisson[iSample] * fRadius.xx), nSMIndex), 0);
        float fDepthSM = GetWSDepth_D3D(fDepthSM_NDC, fNear, fFar);
        float fDeltaDepth = max(0, fDepthFragment - fDepthSM);

        fShadow += ComputeHairShadowAttenuation(fDeltaDepth) * fWeight;

        fTotalWeight += fWeight;
    }
    return fShadow / fTotalWeight;

}




float ComputeIndexedShadow(float3 vPositionWS, float nSMIndex = 0)
{
    if (!bEnableShadow)
        return 1;

    float2 vShadowParams = ShadowParams[int(nSMIndex)].xy;

    float fRadius = vShadowParams.x;
    bool bIsHair = vShadowParams.y > 0;


    if (bIsHair)
        return ComputeShadowHair(vPositionWS, fRadius, nSMIndex);
    else
        return ComputeShadowPoisson10(vPositionWS, fRadius, nSMIndex);
    //return ComputeShadowPCF(vPositionWS, nSMIndex);
}





#line 160 "qHair.sufx"


#line 12 "Effects/D3D11/iShadowedLighting.shl"

StructuredBuffer<uint> nShadowsUpperBuffer;
StructuredBuffer<uint> nShadowsIndexBuffer;

float ComputeLightShadow(int lightIndex, float3 vPositionWS)
{
    float fShadow = 1.0;
    int nShadowsLower = 0;
    if (lightIndex > 0)
        nShadowsLower = nShadowsUpperBuffer[lightIndex - 1];
    int nShadowsUpper = nShadowsUpperBuffer[lightIndex];

    for (int i = nShadowsLower; i < nShadowsUpper; ++i)
    {
        fShadow = min(fShadow, ComputeIndexedShadow(vPositionWS, nShadowsIndexBuffer[i]));
    }
    //if(fShadow != 1.0) fShadow = 0;
    return fShadow;
}


#line 161 "qHair.sufx"


#line 61 "Effects/D3D11/Include/SuLighting.shl"



#line 1157 "Effects/D3DCommon/Include/SuMath.shl"


////////////////////////////////////////////////////////////////////////////
// Convert a quaternion to a matrix.
////////////////////////////////////////////////////////////////////////////
float4x4 SuQuatToMatrix(float4 quat)
{
    /*
          float xx      = quat.x * quat.x;
          float xy      = quat.x * quat.y;
          float xz      = quat.x * quat.z;
          float xw      = quat.x * quat.w;

          float yy      = quat.y * quat.y;
          float yz      = quat.y * quat.z;
          float yw      = quat.y * quat.w;

          float zz      = quat.z * quat.z;
          float zw      = quat.z * quat.w;

          float4x4 mat;
          mat._m00  = 1 - 2 * ( yy + zz );
          mat._m10  =     2 * ( xy - zw );
          mat._m20  =     2 * ( xz + yw );
          mat._m30 = 0.0f;

          mat._m01  =     2 * ( xy + zw );
          mat._m11  = 1 - 2 * ( xx + zz );
          mat._m21  =     2 * ( yz - xw );
          mat._m31 = 0.0f;

          mat._m02  =     2 * ( xz - yw );
          mat._m12  =     2 * ( yz + xw );
          mat._m22 = 1 - 2 * ( xx + yy );
          mat._m32 = 0.0f;

          mat._m03 = 0.0f;
          mat._m13 = 0.0f;
          mat._m23 = 0.0f;
          mat._m33 = 1.0f;
    */

    float x2 = quat.x + quat.x;
    float y2 = quat.y + quat.y;
    float z2 = quat.z + quat.z;

    float xx = quat.x * x2;
    float xy = quat.x * y2;
    float xz = quat.x * z2;
    float yy = quat.y * y2;
    float yz = quat.y * z2;
    float zz = quat.z * z2;
    float wx = quat.w * x2;
    float wy = quat.w * y2;
    float wz = quat.w * z2;

    float4x4 mat;
    mat._m00 = 1.0f - (yy + zz);
    mat._m01 = xy - wz;
    mat._m02 = xz + wy;
    mat._m03 = 0.0f;

    mat._m10 = xy + wz;
    mat._m11 = 1.0f - (xx + zz);
    mat._m12 = yz - wx;
    mat._m13 = 0.0f;

    mat._m20 = xz - wy;
    mat._m21 = yz + wx;
    mat._m22 = 1.0f - (xx + yy);
    mat._m23 = 0.0f;

    mat._m30 = 0.0f;
    mat._m31 = 0.0f;
    mat._m32 = 0.0f;
    mat._m33 = 1.0f;

    return mat;
}

////////////////////////////////////////////////////////////////////////////
// Convert a quaternion/translation pair to a 3x4 transformation matrix
////////////////////////////////////////////////////////////////////////////
float3x4 SuQuatAndTranslationToMatrix(float4 quat, float3 xlate)
{
    float x2 = quat.x + quat.x;
    float y2 = quat.y + quat.y;
    float z2 = quat.z + quat.z;

    float xx = quat.x * x2;
    float xy = quat.x * y2;
    float xz = quat.x * z2;
    float yy = quat.y * y2;
    float yz = quat.y * z2;
    float zz = quat.z * z2;
    float wx = quat.w * x2;
    float wy = quat.w * y2;
    float wz = quat.w * z2;

    float3x4 mat;
    mat._m00 = 1.0f - (yy + zz);
    mat._m01 = xy - wz;
    mat._m02 = xz + wy;
    mat._m03 = xlate.x;

    mat._m10 = xy + wz;
    mat._m11 = 1.0f - (xx + zz);
    mat._m12 = yz - wx;
    mat._m13 = xlate.y;

    mat._m20 = xz - wy;
    mat._m21 = yz + wx;
    mat._m22 = 1.0f - (xx + yy);
    mat._m23 = xlate.z;

    return mat;
}

////////////////////////////////////////////////////////////////////////////
// Converts matrix representation of a rotation to a quaternion.
////////////////////////////////////////////////////////////////////////////
float4 SuMatrixToQuat(float4x4 mat)
{
    float4 quat;
    float fMat[16] = { mat._m00, mat._m10, mat._m20, mat._m30,
                      mat._m01, mat._m11, mat._m21, mat._m31,
                      mat._m02, mat._m12, mat._m22, mat._m32,
                      mat._m03, mat._m13, mat._m23, mat._m33 };

    float tr = mat._m00 + mat._m11 + mat._m22;

    //check the diagonal
    if (tr > 0.0f)
    {
        float s = sqrt(tr + 1.0f);
        quat.w = s / 2.0f;
        s = 0.5f / s;
        quat.x = (mat._m21 - mat._m12) * s;
        quat.y = (mat._m02 - mat._m20) * s;
        quat.z = (mat._m10 - mat._m01) * s;
    }
    else // Diagonal is negative
    {
        // Reworked from C code to eliminate temp registers.
        if (fMat[5] > fMat[0])
        {
            if (fMat[10] > fMat[5])
            {
                float s = sqrt((fMat[10] - (fMat[0] + fMat[5])) + 1.0f);
                quat.z = s * 0.5;
                if (s != 0.0f)
                {
                    s = 0.5f / s;
                }
                quat.w = (fMat[1] - fMat[4]) * s;
                quat.x = (fMat[8] + fMat[2]) * s;
                quat.y = (fMat[9] + fMat[6]) * s;
            }
            else
            {
                float s = sqrt((fMat[5] - (fMat[10] + fMat[0])) + 1.0f);
                quat.y = s * 0.5f;
                if (s != 0.0f)
                {
                    s = 0.5f / s;
                }
                quat.w = (fMat[8] - fMat[2]) * s;
                quat.z = (fMat[6] + fMat[9]) * s;
                quat.x = (fMat[4] + fMat[1]) * s;
            }
        }
        else
        {
            if (fMat[10] > fMat[0])
            {
                float s = sqrt((fMat[10] - (fMat[0] + fMat[5])) + 1.0f);
                quat.z = s * 0.5f;
                if (s != 0.0f)
                {
                    s = 0.5f / s;
                }
                quat.w = (fMat[1] - fMat[4]) * s;
                quat.x = (fMat[8] + fMat[2]) * s;
                quat.y = (fMat[9] + fMat[6]) * s;
            }
            else
            {
                float s = sqrt((fMat[0] - (fMat[5] + fMat[10])) + 1.0f);
                quat.x = s * 0.5f;
                if (s != 0.0f)
                {
                    s = 0.5f / s;
                }
                quat.w = (fMat[6] - fMat[9]) * s;
                quat.y = (fMat[1] + fMat[4]) * s;
                quat.z = (fMat[2] + fMat[8]) * s;
            }
        }
    }
    return normalize(quat);
}

float4 SuMatrixToQuat(float3x3 mat)
{
    float4x4 mMatrix = { float4(mat[0], 0), float4(mat[1], 0), float4(mat[2], 0), float4(0,0,0,1) };
    return SuMatrixToQuat(mMatrix);
}

////////////////////////////////////////////////////////////////////////////
// Converts a quaternion into an axis and angle x, y, and z components
// contain the axis, w contains the angle in degrees.
////////////////////////////////////////////////////////////////////////////
float4 SuQuaternionToGlRotation(float4 quat)
{
    float len = (quat.x * quat.x) + (quat.y * quat.y) + (quat.z * quat.z);
    float4 result;
    if (len > 0.001)
    {
        result.x = quat.x * (1.0f / len);
        result.y = quat.y * (1.0f / len);
        result.z = quat.z * (1.0f / len);
        result.w = degrees(2.0f * acos(quat.w));
    }
    else
    {
        result.x = 0.0;
        result.y = 0.0;
        result.z = 1.0;
        result.w = 0.0;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////
// Creates a quaternion from an axis and angle (in degrees)
////////////////////////////////////////////////////////////////////////////
float4 SuGlRotationToQuaternion(float x, float y, float z, float degrees)
{
    float rad = radians(degrees);
    float tmpf = sqrt(x * x + y * y + z * z);
    if (tmpf != 0.0f)
    {
        float dist = -sin(rad * 0.5f) / tmpf;
        x *= dist;
        y *= dist;
        z *= dist;
    }
    float4 quat;
    quat.x = x;
    quat.y = y;
    quat.z = z;
    quat.w = cos(rad * 0.5f);
    return quat;
}

////////////////////////////////////////////////////////////////////////////
// Multiplies two quaternions
////////////////////////////////////////////////////////////////////////////
float4 SuQuatMul(float4 q1, float4 q2)
{
    float4 tmpq;
    tmpq.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    tmpq.y = (q1.w * q2.y) + (q1.y * q2.w) + (q1.z * q2.x) - (q1.x * q2.z);
    tmpq.z = (q1.w * q2.z) + (q1.z * q2.w) + (q1.x * q2.y) - (q1.y * q2.x);
    tmpq.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    //tmpq = tmpq / (sqrt(tmpq.x*tmpq.x + tmpq.y*tmpq.y + tmpq.z*tmpq.z + tmpq.w*tmpq.w));
    return tmpq;
}

////////////////////////////////////////////////////////////////////////////
// Calculates natural logarithm of a quaternion
////////////////////////////////////////////////////////////////////////////
float4 SuQuatLog(float4 q1)
{
    float length = sqrt(q1.x * q1.x + q1.y * q1.y + q1.z * q1.z);
    length = atan2(length, q1.w);
    float4 q2;
    q2.x = q1.x * length;
    q2.y = q1.y * length;
    q2.z = q1.z * length;
    q2.w = 0.0f;
    return q2;
}

////////////////////////////////////////////////////////////////////////////
// Calculates inverse a quaternion
////////////////////////////////////////////////////////////////////////////
float4 SuQuatInverse(float4 q)
{
    float tmpf = 1.0f / (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    float4 result;
    result.x = -q.x * tmpf;
    result.y = -q.y * tmpf;
    result.z = -q.z * tmpf;
    result.w = q.w * tmpf;
    return result;
}

////////////////////////////////////////////////////////////////////////////
// Calculates exponent of a quaternion
////////////////////////////////////////////////////////////////////////////
float4 SuQuatExp(float4 q1)
{
    float len1 = sqrt(q1.x * q1.x + q1.y * q1.y + q1.z * q1.z);
    float len2;
    if (len1 > 0.0)
        len2 = sin(len1) / len1;
    else
        len2 = 1.0;
    float4 q2;
    q2.x = q1.x * len2;
    q2.y = q1.y * len2;
    q2.z = q1.z * len2;
    q2.w = cos(len1);
    return q2;
}

////////////////////////////////////////////////////////////////////////////
// Smoothly (spherically, shortest path on a quaternion sphere) 
// interpolates between two UNIT quaternion positions
//==========================================================================
// As t goes from 0 to 1, qt goes from p to q.
// slerp(p,q,t) = (p*sin((1-t)*omega) + q*sin(t*omega)) / sin(omega)
////////////////////////////////////////////////////////////////////////////
float4 SuQuatSlerp(float4 from, float4 to, float t)
{
    //Calculate cosine
    float cosom = dot(from, to);

    //Adjust signs (if necessary)
    float4 to1;
    if (cosom < 0.0)
    {
        cosom = -cosom;
        to1[0] = -to.x;
        to1[1] = -to.y;
        to1[2] = -to.z;
        to1[3] = -to.w;
    }
    else
    {
        to1[0] = to.x;
        to1[1] = to.y;
        to1[2] = to.z;
        to1[3] = to.w;
    }

    //Calculate coefficients
    float scale0;
    float scale1;
    if ((1.0f - cosom) > 0.001f)
    {
        //Standard case (slerp)
        float omega = acos(cosom);
        float sinom = sin(omega);
        scale0 = sin((1.0f - t) * omega) / sinom;
        scale1 = sin(t * omega) / sinom;
    }
    else
    {
        //"from" and "to" quaternions are very close
        //... so we can do a linear interpolation
        scale0 = 1.0f - t;
        scale1 = t;
    }

    // calculate final values
    float4 result;
    result.x = (scale0 * from.x) + (scale1 * to1[0]);
    result.y = (scale0 * from.y) + (scale1 * to1[1]);
    result.z = (scale0 * from.z) + (scale1 * to1[2]);
    result.w = (scale0 * from.w) + (scale1 * to1[3]);
    return result;
}

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
float4 SuQuatSlerpShortestPath(float4 from, float4 to, float t)
{
    float4 result;
    if (dot(from, to) < 0.0f)
    {
        float4 tmpFrom = { -from.x, -from.y, -from.z, -from.w };
        result = SuQuatSlerp(tmpFrom, to, t);
    }
    else
    {
        result = SuQuatSlerp(from, to, t);
    }
    return normalize(result);
}

////////////////////////////////////////////////////////////////////////////
// Cross product of 3 quaternions: result = (q1 X q2 X q3)
//
// This is done by creating the matrix:
//      i   j   k   m
//      x1  x2  x3  x4
//      y1  y2  y3  y4
//      z1  z2  z3  z4
//      w1  w2  w3  w4
// and solving for i, j, k, and m. These 4 values compose the resulting
// Quaternion: result = (i, j, k, m)
////////////////////////////////////////////////////////////////////////////
float4 SuQuatAxBxC(float4 q1, float4 q2, float4 q3)
{
    float4 result;
    float3x3 det; //Determinant to pass to function

    // result->x
    det[0][0] = q1.y;
    det[0][1] = q1.z;
    det[0][2] = q1.w;

    det[1][0] = q2.y;
    det[1][1] = q2.z;
    det[1][2] = q2.w;

    det[2][0] = q3.y;
    det[2][1] = q3.z;
    det[2][2] = q3.w;

    result.x = determinant(det);

    // result->y
    det[0][0] = q1.x;
    det[0][1] = q1.z;
    det[0][2] = q1.w;

    det[1][0] = q2.x;
    det[1][1] = q2.z;
    det[1][2] = q2.w;

    det[2][0] = q3.x;
    det[2][1] = q3.z;
    det[2][2] = q3.w;

    result.y = -determinant(det);

    // result->z
    det[0][0] = q1.x;
    det[0][1] = q1.y;
    det[0][2] = q1.w;

    det[1][0] = q2.x;
    det[1][1] = q2.y;
    det[1][2] = q2.w;

    det[2][0] = q3.x;
    det[2][1] = q3.y;
    det[2][2] = q3.w;

    result.z = determinant(det);

    // result->w //
    det[0][0] = q1.x;
    det[0][1] = q1.y;
    det[0][2] = q1.z;

    det[1][0] = q2.x;
    det[1][1] = q2.y;
    det[1][2] = q2.z;

    det[2][0] = q3.x;
    det[2][1] = q3.y;
    det[2][2] = q3.z;

    result.w = -determinant(det);
    return result;
}
#line 63 "Effects/D3D11/Include/SuLighting.shl"


#line 871 "Effects/D3DCommon/Include/SuMath.shl"

////////////////////////////////////////////////////////////////////////////
// Returns a full 4x4 identity matrix
////////////////////////////////////////////////////////////////////////////
float4x4 SuMakeIdentityMatrix(void)
{
    float4x4 mat = { 1.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 1.0f };
    return mat;
}

////////////////////////////////////////////////////////////////////////////
// Create a translation matrix given the x, y, and z translation components
////////////////////////////////////////////////////////////////////////////
float4x4 SuMakeTranslationMatrix(float x, float y, float z)
{
    float4x4 mat = SuMakeIdentityMatrix();
    mat._m03 = x;
    mat._m13 = y;
    mat._m23 = z;
    return mat;
}

////////////////////////////////////////////////////////////////////////////
// Create a scale matrix given the x, y, and z scaling components
////////////////////////////////////////////////////////////////////////////
float4x4 SuMakeScaleMatrix(float x, float y, float z)
{
    float4x4 mat = SuMakeIdentityMatrix();
    mat._m00 = x;
    mat._m11 = y;
    mat._m22 = z;
    return mat;
}

////////////////////////////////////////////////////////////////////////////
// Create a rotation matrix that rotates a vector called "from" into another
// vector called "to".
// source: http://www.cs.lth.se/home/Tomas_Akenine_Moller/code/fromtorot.txt
////////////////////////////////////////////////////////////////////////////
float3x3 SuMakeFromToRotationMatrix(float3 vFrom, float3 vTo)
{
    float3x3 mResult;
    float3 vV = cross(vFrom, vTo);
    float fE = dot(vFrom, vTo);
    float fF = abs(fE);

    if (fF > 0.999) // vFrom and vTo almost parallel
    {
        float3 vX = abs(vFrom);

        if (vX.x < vX.y)
        {
            if (vX.x < vX.z)
            {
                vX = float3(1.0, 0.0, 0.0);
            }
            else
            {
                vX = float3(0.0, 0.0, 1.0);
            }
        }
        else
        {
            if (vX.y < vX.z)
            {
                vX = float3(0.0, 1.0, 0.0);
            }
            else
            {
                vX = float3(0.0, 0.0, 1.0);
            }
        }

        float3 vU = vX - vFrom;
        float3 vV = vX - vTo;

        float fC1 = 2.0 / dot(vU, vU);
        float fC2 = 2.0 / dot(vV, vV);
        float fC3 = fC1 * fC2 * dot(vU, vV);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                mResult[j][i] = -fC1 * vU[i] * vU[j]
                    - fC2 * vV[i] * vV[j]
                    + fC3 * vV[i] * vU[j];
            }

            mResult[i][i] += 1.0;
        }

    }
    else
    {
        float fH = 1.0 / (1.0 + fE);

        mResult[0][0] = fE + fH * vV.x * vV.x;
        mResult[1][0] = fH * vV.x * vV.y + vV.z;
        mResult[2][0] = fH * vV.x * vV.z - vV.y;

        mResult[0][1] = fH * vV.x * vV.y - vV.z;
        mResult[1][1] = fE + fH * vV.y * vV.y;
        mResult[2][1] = fH * vV.y * vV.z + vV.x;

        mResult[0][2] = fH * vV.x * vV.z + vV.y;
        mResult[1][2] = fH * vV.y * vV.z - vV.x;
        mResult[2][2] = fE + fH * vV.z * vV.z;
    }

    return mResult;
}

////////////////////////////////////////////////////////////////////////////
// Same as above, but skips the test to see if the vectors are parallel
////////////////////////////////////////////////////////////////////////////
float3x3 SuMakeFromToRotationMatrixFast(float3 vFrom, float3 vTo)
{
    float3x3 mResult;
    float3 vV = cross(vFrom, vTo);
    float fE = dot(vFrom, vTo);
    float fH = 1.0 / (1.0 + fE);

    mResult[0][0] = fE + fH * vV.x * vV.x;
    mResult[1][0] = fH * vV.x * vV.y + vV.z;
    mResult[2][0] = fH * vV.x * vV.z - vV.y;

    mResult[0][1] = fH * vV.x * vV.y - vV.z;
    mResult[1][1] = fE + fH * vV.y * vV.y;
    mResult[2][1] = fH * vV.y * vV.z + vV.x;

    mResult[0][2] = fH * vV.x * vV.z + vV.y;
    mResult[1][2] = fH * vV.y * vV.z - vV.x;
    mResult[2][2] = fE + fH * vV.z * vV.z;


    return mResult;
}

////////////////////////////////////////////////////////////////////////////
// 
// Calculate the rotation matrix that corresponds to the rotation specified
// by an arbitrary axis whose components determine the direction of the 
// axis and whose length indicates the amount of rotation (this is what 
// physicists call the infinitesimal angular displacement)
//
// Source: Robot Manipulators: Mathematics, Programming and Control
//         by Richard Paul, pg. 28
//
//
//  This is the matrix m:
//
// WARNING: vAxis is assumed to be unit length!
// 
// --                                                                 --
//| kx*kx*vers0 + cos0     ky*kx*vers0 - kz*sin0  kz*kx*vers0 + ky*sin0 |
//| kx*ky*vers0 + kz*sin0  ky*ky*vers0 + kz*cos0  kz*ky*vers0 - kx*sin0 |
//| kx*kz*vers0 - ky*sin0  ky*kz*vers0 + kx*sin0  kz*kz*vers0 + cos0    |
// --                                                                 --
////////////////////////////////////////////////////////////////////////////
float3x3 SuArbitraryAxisToMatrix(float3 vAxis, float fRadians)
{
    float3x3 mTransform;

    if (abs(fRadians) < 0.001)
    {
        mTransform = SuMakeIdentityMatrix();
    }
    else
    {
        // Calculate angles
        float fSin0 = sin(fRadians);
        float fCos0 = cos(fRadians);
        float fVers0 = 1.0 - fCos0;

        // Calculate commonly used values
        float3 vVers0 = vAxis * fVers0;
        float3 vSin0 = vAxis * fSin0;

        // Calculate matrix elements (see above descriptions) //
        mTransform[0] = vAxis * vVers0.xxx + float3(fCos0, -vSin0.z, vSin0.y);
        mTransform[1] = vAxis * vVers0.yyy + float3(vSin0.z, fCos0, -vSin0.x);
        mTransform[2] = vAxis * vVers0.zzz + float3(-vSin0.y, vSin0.x, fCos0);
    }

    return mTransform;
}


////////////////////////////////////////////////////////////////////////////
// Multiply two 4x4 matrices.
////////////////////////////////////////////////////////////////////////////
float4x4 SuMult44x44(float4x4 m1, float4x4 m2)
{
    return mul(m1, m2);
}

////////////////////////////////////////////////////////////////////////////
// Multiply a 4x4 matrix with a 3 space vector
////////////////////////////////////////////////////////////////////////////
float4 SuMult44x41(float4x4 mat, float4 vec)
{
    return mul(mat, vec);
}

////////////////////////////////////////////////////////////////////////////
// Multiply a 4x4 matrix with a 3 space vector
////////////////////////////////////////////////////////////////////////////
float3 SuMult44x31(float4x4 mat, float3 vec)
{
    return mul(mat, float4(vec, 1));
}

////////////////////////////////////////////////////////////////////////////
// Multiply the 3x3 portion of the 4x4 matrix with a 3 space vector
////////////////////////////////////////////////////////////////////////////
float3 SuMult33x31(float4x4 mat, float3 vec)
{
    return mul(mat, float4(vec, 0));
}

////////////////////////////////////////////////////////////////////////////
// Multiply a 3x3 matrix with a 3 space vector
////////////////////////////////////////////////////////////////////////////
float3 SuMult33x31(float3x3 mat, float3 vec)
{
    return mul(mat, vec);
}

////////////////////////////////////////////////////////////////////////////
// Multiply the 3x3 portion of the 4x4 matrix with a 4 space vector
////////////////////////////////////////////////////////////////////////////
float4 SuMult33x41(float4x4 mat, float4 vec)
{
    float3 v = mul(mat, float4(vec.xyz, 0));
    return float4 (v, 1);
}

////////////////////////////////////////////////////////////////////////////
// Multiply a 3x3 matrix with a 4 space vector
////////////////////////////////////////////////////////////////////////////
float4 SuMult33x41(float3x3 mat, float4 vec)
{
    float3 v = mul(mat, vec.xyz);
    return float4 (v, 1);
}

////////////////////////////////////////////////////////////////////////////
// Transpose a matrix.
////////////////////////////////////////////////////////////////////////////
float4x4 SuTransposeMatrix(float4x4 mat)
{
    return transpose(mat);
}

////////////////////////////////////////////////////////////////////////////
// Cheap(er) invert matrix. Only works for orthonormal matrices with no
// shearing/scaling.
////////////////////////////////////////////////////////////////////////////
float4x4 SuRigidInvertMatrix(float4x4 inMat)
{
    // Rotations are just the transpose.
    float4x4 mat = transpose(inMat);

    // Clear shearing terms
    mat[3] = float4(0, 0, 0, 1);

    // Translation is minus the dot of tranlation and rotations
    float4 v = float4(inMat._m03, inMat._m13, inMat._m23, 0);
    mat._m03 = -dot(v, float4 (inMat._m00, inMat._m01, inMat._m02, 0));
    mat._m13 = -dot(v, float4 (inMat._m10, inMat._m11, inMat._m12, 0));
    mat._m23 = -dot(v, float4 (inMat._m20, inMat._m21, inMat._m22, 0));

    return mat;
}

#line 64 "Effects/D3D11/Include/SuLighting.shl"


// =============================================================================================================================
static const int POINT_LIGHT = 0;
static const int DIRECTIONAL_LIGHT = 1;
static const int SPOT_LIGHT = 2;
static const int VOLUME_LIGHT = 3;

static const int VOLUME_LIGHT_SPHERE = 1;
static const int VOLUME_LIGHT_CONE = 3;

uniform int nNumLights;
uniform int nLightShape[SU_MAX_LIGHTS];
uniform int nLightIndex[SU_MAX_LIGHTS];
uniform float fLightIntensity[SU_MAX_LIGHTS];
uniform float3 vLightPosWS[SU_MAX_LIGHTS];
uniform float3 vLightDirWS[SU_MAX_LIGHTS];
uniform float3 vLightColor[SU_MAX_LIGHTS];
uniform float3 vLightConeAngles[SU_MAX_LIGHTS];
uniform float3 vLightScaleWS[SU_MAX_LIGHTS];
uniform float4 vLightParams[SU_MAX_LIGHTS];
uniform float4 vLightOrientationWS[SU_MAX_LIGHTS];


// =============================================================================================================================
float SuComputeSpotLightAttenuation(float3 vLightVecWS, int nLightID)
{
    // spotlight falloff
    float fCosLightAngle = dot(-vLightVecWS, vLightDirWS[nLightID]);
    float fAttenuation = smoothstep(vLightConeAngles[nLightID].x, vLightConeAngles[nLightID].y, fCosLightAngle);

#ifndef SU_NO_LIGHT_DROPOFF
    // dropoff
    fAttenuation *= SuPow(fCosLightAngle, vLightConeAngles[nLightID].z);
#endif

    return fAttenuation;
}

// =============================================================================================================================
float SuComputeLightFalloff(float fLightDistance, int nLightID)
{
#ifdef SU_NO_LIGHT_FALLOFF
    return 1.0;
#else
    return SuPow(fLightDistance, -vLightParams[nLightID].y); // lightParams.y is falloff exponent
#endif
}

// =============================================================================================================================
bool SuIsVolume(int nLightID)
{
    return (vLightParams[nLightID].x == VOLUME_LIGHT);
}


// =============================================================================================================================
float3 SuComputeVolumeAmbient(float3 vPositionWS, int nLightID)
{
    // Inverse Transform
    float4x4 mInverseRotation = SuTransposeMatrix(SuQuatToMatrix(vLightOrientationWS[nLightID]));
    float4x4 mInverseScale = SuMakeScaleMatrix(1.0f / vLightScaleWS[nLightID].x,
        1.0f / vLightScaleWS[nLightID].y,
        1.0f / vLightScaleWS[nLightID].z);
    float4x4 mInverseTransform = SuMult44x44(mInverseScale, mInverseRotation);

    // Compute attenuation
    float3 vPositionLS = SuMult44x31(mInverseTransform, vPositionWS - vLightPosWS[nLightID]);
    float fDist = length(vPositionLS);

    SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_SPHERE)
    {
        if (fDist < 1 + SU_EPSILON)
            return fLightIntensity[nLightID] * vLightColor[nLightID];
    }

        else SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_CONE)
        {
            float fHeight = -vPositionLS.y;
            if ((0.0f - SU_EPSILON < fHeight) && (fHeight < 1.0f + SU_EPSILON))
            {
                float fDist = length(float2(vPositionLS.x, vPositionLS.z));
                if (fDist < fHeight)
                    return fLightIntensity[nLightID] * vLightColor[nLightID];
            }
            }

            return float3(0, 0, 0);
}



// =============================================================================================================================

float SuComputeVolumeLightFalloff(float3 vPositionWS, int nLightID)
{
    // Inverse Transform
    float4x4 mInverseRotation = SuTransposeMatrix(SuQuatToMatrix(vLightOrientationWS[nLightID]));
    float4x4 mInverseScale = SuMakeScaleMatrix(1.0f / vLightScaleWS[nLightID].x,
        1.0f / vLightScaleWS[nLightID].y,
        1.0f / vLightScaleWS[nLightID].z);
    float4x4 mInverseTransform = SuMult44x44(mInverseScale, mInverseRotation);

    // Compute attenuation
    float3 vPositionLS = SuMult44x31(mInverseTransform, vPositionWS - vLightPosWS[nLightID]);
    float fDist = length(vPositionLS);
    float fAttenuation = 1.0;

    SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_SPHERE)
    {
        if (fDist < 1 + SU_EPSILON)
        {
            fAttenuation = 1 - fDist;
        }
    }

        else SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_CONE)
        {
            float fHeight = -vPositionLS.y;
            if ((0.0f - SU_EPSILON < fHeight) && (fHeight < 1.0f + SU_EPSILON))
            {
                float fDist = length(float2(vPositionLS.x, vPositionLS.z));
                if (fDist < fHeight)
                {
                    fAttenuation = 1 - (fDist / fHeight);
                }
            }
            }

            return fAttenuation;
}

// =============================================================================================================================
float3 SuGetVectorToLightPoint(float3 vPositionWS, int nLightID)
{
    return normalize(vLightPosWS[nLightID] - vPositionWS);
}

// =============================================================================================================================
float3 SuGetVectorToLightDirectional(int nLightID)
{
    return -vLightDirWS[nLightID];
}

// =============================================================================================================================
float3 SuGetVectorToLight(float3 vPositionWS, int nLightID)
{
    SU_CRAZY_IF(vLightParams[nLightID].x != DIRECTIONAL_LIGHT)
    {
        return SuGetVectorToLightPoint(vPositionWS, nLightID);
    }
      else
      {
          return SuGetVectorToLightDirectional(nLightID);
          }
}

// =============================================================================================================================
// Reduces light to a direction and scaled color (taking attenuation and light strenght into account)
void SuGetLightLocal(float3 vPositionWS, int nLightID, out float3 vLightOutDirWS, out float3 cAttenulatedColor)
{
    float fIntensity = fLightIntensity[nLightID];
    SU_CRAZY_IF(vLightParams[nLightID].x == DIRECTIONAL_LIGHT)
    {
        vLightOutDirWS = -vLightDirWS[nLightID];
        cAttenulatedColor = fIntensity * vLightColor[nLightID];
        return;
    }

    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    vLightOutDirWS = normalize(vLightVecWS);

    SU_CRAZY_IF(vLightParams[nLightID].x == VOLUME_LIGHT)
    {
        fIntensity *= SuComputeVolumeLightFalloff(vPositionWS, nLightID);
    }
      else
      {
          fIntensity *= SuComputeLightFalloff(length(vLightVecWS), nLightID);
          }

          SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
          {
              fIntensity *= SuComputeSpotLightAttenuation(normalize(vLightVecWS), nLightID);
          }

          cAttenulatedColor = fIntensity * vLightColor[nLightID];
          return;
}

// =============================================================================================================================
float3 SuGetLightColor(float3 vPositionWS, int nLightID)
{
    float3 cColor;
    SU_CRAZY_IF(vLightParams[nLightID].x == DIRECTIONAL_LIGHT)
    {
        cColor = vLightColor[nLightID];
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
      {
          float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
          cColor = vLightColor[nLightID] * SuComputeLightFalloff(length(vLightVecWS), nLightID);
          cColor *= SuComputeSpotLightAttenuation(normalize(vLightVecWS), nLightID);
          }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
          cColor = vLightColor[nLightID] * SuComputeLightFalloff(length(vLightVecWS), nLightID);
          }

          return cColor;
}


// =============================================================================================================================
// DIFFUSE LIGHTING
// =============================================================================================================================

// =============================================================================================================================
float3 SuComputeDiffusePointLight(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    // we could do the whole computation conditionally based on the diffuse mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fDiffuse = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fDiffuse *= saturate(dot(vLightVecWS, vNormalWS));

    return fDiffuse * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeDiffuseSpotLight(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    // we could do the whole computation conditionally based on the diffuse mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fDiffuse = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fDiffuse *= saturate(dot(vLightVecWS, vNormalWS));
    fDiffuse *= SuComputeSpotLightAttenuation(vLightVecWS, nLightID);

    return fDiffuse * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeDiffuseDirectionalLight(float3 vNormalWS, int nLightID)
{
    // falloff doesn't make any sense for directional light sources
    float fDiffuse = saturate(dot(-vLightDirWS[nLightID], vNormalWS));

    return fDiffuse * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeDiffuse(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    float3 cDiffuse;

    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        cDiffuse = SuComputeDiffuseSpotLight(vPositionWS, vNormalWS, nLightID);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          cDiffuse = SuComputeDiffusePointLight(vPositionWS, vNormalWS, nLightID);
          }
      else
      {
          cDiffuse = SuComputeDiffuseDirectionalLight(vNormalWS, nLightID);
          }

          return cDiffuse;
}

// =============================================================================================================================
// SPECULAR LIGHTING
// =============================================================================================================================

// =============================================================================================================================
//            PHONG LIGHTING
// =============================================================================================================================   
// =============================================================================================================================
float3 SuComputeSpecularPointLightPhong(float3 vPositionWS, float3 vReflectedViewWS, float fSpecExp, int nLightID)
{
    // we could do the whole computation conditionally based on the specular mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fSpec = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fSpec *= SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);

    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}

// =============================================================================================================================
float3 SuComputeSpecularSpotLightPhong(float3 vPositionWS, float3 vReflectedViewWS, float fSpecExp, int nLightID)
{
    // we could do the whole computation conditionally based on the specular mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fSpec = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fSpec *= SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);
    fSpec *= SuComputeSpotLightAttenuation(vLightVecWS, nLightID);

    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}

// =============================================================================================================================
float3 SuComputeSpecularDirectionalLightPhong(float3 vReflectedViewWS, float fSpecExp, int nLightID)
{
    // we could do the whole computation conditionally based on the specular mask
    float fSpec = SuPow(saturate(dot(-vLightDirWS[nLightID], vReflectedViewWS)), fSpecExp);

    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}

// =============================================================================================================================
float3 SuComputeSpecularPhong(float3 vPositionWS, float3 vReflectedViewWS, float fSpecExp, int nLightID)
{
    float3 cSpec;

    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        cSpec = SuComputeSpecularSpotLightPhong(vPositionWS, vReflectedViewWS, fSpecExp, nLightID);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          cSpec = SuComputeSpecularPointLightPhong(vPositionWS, vReflectedViewWS, fSpecExp, nLightID);
          }
      else
      {
          cSpec = SuComputeSpecularDirectionalLightPhong(vReflectedViewWS, fSpecExp, nLightID);
          }

          return cSpec;
}
// =============================================================================================================================
//            TORRANCE-SPARROW LIGHTING
// ============================================================================================================================= 
//===========================================================================================================
// Torrance-Sparrow with Blinn microfacet distribution (aka Blinn-Phong)
// exponent = phong exponent for highlight
//===========================================================================================================   
float GeometryDist(float NdotH, float NdotV, float VdotH, float NdotL)
{
    return min(1.0, min(2 * (NdotH) * (NdotV) / (VdotH), 2 * (NdotH) * (NdotL) / (VdotH)));
}

float MicrofacetDist(float exponent, float NdotH)
{
    return SuPow(saturate(NdotH), exponent);
}
// =============================================================================================================================
float3  SuComputeSpecularPointLightTSB(float3 vPositionWS, float3 vNormalWS, float3 vViewWS,
    float fSpecExp, int nLightID)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    // we could do the whole computation conditionally based on the specular mask
    float3 vLightWS = vLightPosWS[nLightID] - vPositionWS;
    float fSpec = SuComputeLightFalloff(length(vLightWS), nLightID);
    vLightWS = normalize(vLightWS);

    // Calculate half vector between light and view
    float3 vHalfWS = (vLightWS + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, vLightWS);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    // Calculate specular and diffuse
    fSpec *= D * G * F / NdotV;
    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}
// =============================================================================================================================
float3  SuComputeSpecularSpotLightTSB(float3 vPositionWS, float3 vNormalWS, float3 vViewWS,
    float fSpecExp, int nLightID)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    // we could do the whole computation conditionally based on the specular mask
    float3 vLightWS = vLightPosWS[nLightID] - vPositionWS;
    float fSpec = SuComputeLightFalloff(length(vLightWS), nLightID);
    vLightWS = normalize(vLightWS);

    // Calculate half vector between light and view
    float3 vHalfWS = (vLightWS + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, vLightWS);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    // Calculate specular
    fSpec *= D * G * F / NdotV;
    fSpec *= SuComputeSpotLightAttenuation(vLightWS, nLightID);
    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}

// =============================================================================================================================
float3 SuComputeSpecularDirectionalLightTSB(float3 vNormalWS, float3 vViewWS, float fSpecExp, int nLightID)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    // Calculate half vector between light and view
    float3 vHalfWS = (-vLightDirWS[nLightID] + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, -vLightDirWS[nLightID]);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    // we could do the whole computation conditionally based on the specular mask
    float fSpec = D * G * F / NdotV;

    return fSpec * vLightColor[nLightID] * vLightParams[nLightID].w;   // vLightParams.w is the specular mask
}

// =============================================================================================================================
float3 SuComputeSpecularTSB(float3 vPositionWS, float3 vNormalWS, float3 vViewWS, float fSpecExp, int nLightID)
{
    float3 cSpec;

    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        cSpec = SuComputeSpecularSpotLightTSB(vPositionWS, vNormalWS, vViewWS, fSpecExp, nLightID);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          cSpec = SuComputeSpecularPointLightTSB(vPositionWS, vNormalWS, vViewWS, fSpecExp, nLightID);
          }
      else
      {
          cSpec = SuComputeSpecularDirectionalLightTSB(vNormalWS, vViewWS, fSpecExp, nLightID);
          }

          return cSpec;
}
// =============================================================================================================================
// COMBINED DIFFUSE SPECULAR
// ============================================================================================================================= 
// =============================================================================================================================
//===========================================================================================================
//       COMBINED PHONG
//===========================================================================================================      
void SuComputeDiffuseSpecPointLightPhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fFalloff = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    float2 vDS;
    vDS.x = saturate(dot(vLightVecWS, vNormalWS));
    vDS.y = SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);
    vDS *= vLightParams[nLightID].zw;
    vDS *= fFalloff;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

// =============================================================================================================================
void SuComputeDiffuseSpecSpotLightPhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fFalloff = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    float2 vDS;
    vDS.x = saturate(dot(vLightVecWS, vNormalWS));
    vDS.y = SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);
    vDS *= SuComputeSpotLightAttenuation(vLightVecWS, nLightID);
    vDS *= vLightParams[nLightID].zw;
    vDS *= fFalloff;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

// =============================================================================================================================
void SuComputeDiffuseSpecDirectionalLightPhong(float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    float2 vDS;
    vDS.x = saturate(dot(-vLightDirWS[nLightID], vNormalWS));
    vDS.y = SuPow(saturate(dot(-vLightDirWS[nLightID], vReflectedViewWS)), fSpecExp);
    vDS *= vLightParams[nLightID].zw;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

void SuComputeDiffuseSpecVolumeLightSpherePhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    // Inverse Transform
    float4x4 mInverseRotation = SuTransposeMatrix(SuQuatToMatrix(vLightOrientationWS[nLightID]));
    float4x4 mInverseScale = SuMakeScaleMatrix(1.0f / vLightScaleWS[nLightID].x,
        1.0f / vLightScaleWS[nLightID].y,
        1.0f / vLightScaleWS[nLightID].z);
    float4x4 mInverseTransform = SuMult44x44(mInverseScale, mInverseRotation);

    // Compute attenuation
    float3 vPositionLS = SuMult44x31(mInverseTransform, vPositionWS - vLightPosWS[nLightID]);
    float fDist = length(vPositionLS);
    float fAttenuation = 0;

    if (fDist < 1 + SU_EPSILON)
    {
        fAttenuation = 1 - fDist;
    }

    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    vLightVecWS = normalize(vLightVecWS);
    float2 vDS;
    vDS.x = saturate(dot(vLightVecWS, vNormalWS));
    vDS.y = SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);
    vDS *= vLightParams[nLightID].zw;
    vDS *= fAttenuation;

    cDiffuse = vDS.x * vLightColor[nLightID] * fLightIntensity[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID] * fLightIntensity[nLightID];
}

void SuComputeDiffuseSpecVolumeLightConePhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    // Inverse Transform
    float4x4 mInverseRotation = SuTransposeMatrix(SuQuatToMatrix(vLightOrientationWS[nLightID]));
    float4x4 mInverseScale = SuMakeScaleMatrix(1.0f / vLightScaleWS[nLightID].x,
        1.0f / vLightScaleWS[nLightID].y,
        1.0f / vLightScaleWS[nLightID].z);
    float4x4 mInverseTransform = SuMult44x44(mInverseScale, mInverseRotation);

    // Compute attenuation
    float3 vPositionLS = SuMult44x31(mInverseTransform, vPositionWS - vLightPosWS[nLightID]);

    float fAttenuation = 0;
    float fHeight = -vPositionLS.y;
    if ((0.0f - SU_EPSILON < fHeight) && (fHeight < 1.0f + SU_EPSILON))
    {
        float fDist = length(float2(vPositionLS.x, vPositionLS.z));
        if (fDist < fHeight)
        {
            fAttenuation = 1 - (fDist / fHeight);
        }
    }

    float3 vLightVecWS = vPositionWS - vLightPosWS[nLightID];
    vLightVecWS = normalize(vLightVecWS);
    float2 vDS;
    vDS.x = saturate(dot(vLightVecWS, vNormalWS));
    vDS.y = SuPow(saturate(dot(vLightVecWS, vReflectedViewWS)), fSpecExp);
    vDS *= vLightParams[nLightID].zw;
    vDS *= fAttenuation;

    cDiffuse = vDS.x * vLightColor[nLightID] * fLightIntensity[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID] * fLightIntensity[nLightID];
}

void SuComputeDiffuseSpecVolumeLightPhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_SPHERE)
    {
        SuComputeDiffuseSpecVolumeLightSpherePhong(vPositionWS, vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
    }
      else SU_CRAZY_IF(nLightShape[nLightID] == VOLUME_LIGHT_CONE)
      {
          SuComputeDiffuseSpecVolumeLightConePhong(vPositionWS, vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
      else
      {
          //not supported
          cDiffuse = float3(1, 0, 1);
          cSpecular = float3(0, 0, 0);
          }
}

// =============================================================================================================================
void SuComputeDiffuseSpecPhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vReflectedViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        SuComputeDiffuseSpecSpotLightPhong(vPositionWS, vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          SuComputeDiffuseSpecPointLightPhong(vPositionWS, vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
      else SU_CRAZY_IF(vLightParams[nLightID].x == VOLUME_LIGHT)
      {
          SuComputeDiffuseSpecVolumeLightPhong(vPositionWS, vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
      else
      {
          SuComputeDiffuseSpecDirectionalLightPhong(vNormalWS, vReflectedViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
}

// =============================================================================================================================
void SuComputeDiffuseSpecPhong(float3 vPositionWS,
    float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    float3 vReflWS = SuReflect(vViewWS, vNormalWS);

    cDiffuse = float3(0, 0, 0);
    cSpecular = float3(0, 0, 0);

    float3 cDiff;
    float3 cSpec;

    for (int i = 0; i < nNumLights; i++)
    {
        SuComputeDiffuseSpecPhong(vPositionWS, vNormalWS, vReflWS, fSpecExp, i, cDiff, cSpec);
        cDiffuse += cDiff;
        cSpecular += cSpec;
    }
}

//===========================================================================================================
//       COMBINED TSB
//===========================================================================================================      
void SuComputeDiffuseSpecPointLightTSB(float3 vPositionWS,
    float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fFalloff = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);

    // Calculate half vector between light and view
    float3 vHalfWS = (vLightVecWS + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, vLightVecWS);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    float2 vDS;
    vDS.x = saturate(NdotL);
    vDS.y = D * G * F / NdotV;
    vDS *= vLightParams[nLightID].zw;
    vDS *= fFalloff;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

// =============================================================================================================================
void SuComputeDiffuseSpecSpotLightTSB(float3 vPositionWS,
    float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fFalloff = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);

    // Calculate half vector between light and view
    float3 vHalfWS = (vLightVecWS + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, vLightVecWS);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    float2 vDS;
    vDS.x = saturate(NdotL);
    vDS.y = D * G * F / NdotV;
    vDS *= SuComputeSpotLightAttenuation(vLightVecWS, nLightID);
    vDS *= vLightParams[nLightID].zw;
    vDS *= fFalloff;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

// =============================================================================================================================
void SuComputeDiffuseSpecDirectionalLightTSB(float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    // This is just a test.. may want to hardcode it later
    float fR = 1.0 / fSpecExp;

    // Calculate half vector between light and view
    float3 vHalfWS = (-vLightDirWS[nLightID] + vViewWS);
    vHalfWS /= length(vHalfWS);

    // Calculate all the angles we need
    float NdotH = dot(vNormalWS, vHalfWS);
    float NdotV = dot(vNormalWS, vViewWS);
    float VdotH = dot(vViewWS, vHalfWS);
    float NdotL = dot(vNormalWS, -vLightDirWS[nLightID]);

    // Calculate microfacet distribution, masking term, and fresnel
    float G = GeometryDist(NdotH, NdotV, VdotH, NdotL);
    float D = MicrofacetDist(fSpecExp, NdotH);
    float F = SchlickFresnel(NdotV, fR);

    float2 vDS;
    vDS.x = saturate(NdotL);
    vDS.y = D * G * F / NdotV;
    vDS *= vLightParams[nLightID].zw;

    cDiffuse = vDS.x * vLightColor[nLightID];
    cSpecular = vDS.y * vLightColor[nLightID];
}

// =============================================================================================================================
void SuComputeDiffuseSpecTSB(float3 vPositionWS,
    float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    int nLightID,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        SuComputeDiffuseSpecSpotLightTSB(vPositionWS, vNormalWS, vViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          SuComputeDiffuseSpecPointLightTSB(vPositionWS, vNormalWS, vViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
      else
      {
          SuComputeDiffuseSpecDirectionalLightTSB(vNormalWS, vViewWS, fSpecExp, nLightID, cDiffuse, cSpecular);
          }
}

// =============================================================================================================================
void SuComputeDiffuseSpecTSB(float3 vPositionWS,
    float3 vNormalWS,
    float3 vViewWS,
    float fSpecExp,
    out float3 cDiffuse,
    out float3 cSpecular)
{
    float3 vReflWS = SuReflect(vViewWS, vNormalWS);
    cDiffuse = float3(0, 0, 0);
    cSpecular = float3(0, 0, 0);

    float3 cDiff;
    float3 cSpec;

    for (int i = 0; i < nNumLights; i++)
    {
        SuComputeDiffuseSpecTSB(vPositionWS, vNormalWS, vReflWS, fSpecExp, i, cDiff, cSpec);
        cDiffuse += cDiff;
        cSpecular += cSpec;
    }
}
// =============================================================================================================================
// TERMINATOR LIGHTING
// =============================================================================================================================

// =============================================================================================================================
float3 SuComputeTerminatorPointLight(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    // we could do the whole computation conditionally based on the diffuse mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fTerminator = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fTerminator *= (1.0 - abs(dot(vLightVecWS, vNormalWS)));

    return fTerminator * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeTerminatorSpotLight(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    // we could do the whole computation conditionally based on the diffuse mask
    float3 vLightVecWS = vLightPosWS[nLightID] - vPositionWS;
    float fTerminator = SuComputeLightFalloff(length(vLightVecWS), nLightID);
    vLightVecWS = normalize(vLightVecWS);
    fTerminator *= (1.0 - abs(dot(vLightVecWS, vNormalWS)));
    fTerminator *= SuComputeSpotLightAttenuation(vLightVecWS, nLightID);

    return fTerminator * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeTerminatorDirectionalLight(float3 vNormalWS, int nLightID)
{
    // falloff doesn't make any sense for directional light sources
    float fTerminator = (1.0 - abs(dot(-vLightDirWS[nLightID], vNormalWS)));

    return fTerminator * vLightColor[nLightID] * vLightParams[nLightID].z;   // vLightParams.z is the diffuse mask
}

// =============================================================================================================================
float3 SuComputeTerminator(float3 vPositionWS, float3 vNormalWS, int nLightID)
{
    float3 cTerminator;

    SU_CRAZY_IF(vLightParams[nLightID].x == SPOT_LIGHT)
    {
        cTerminator = SuComputeTerminatorSpotLight(vPositionWS, vNormalWS, nLightID);
    }
      else SU_CRAZY_IF(vLightParams[nLightID].x == POINT_LIGHT)
      {
          cTerminator = SuComputeTerminatorPointLight(vPositionWS, vNormalWS, nLightID);
          }
      else
      {
          cTerminator = SuComputeTerminatorDirectionalLight(vNormalWS, nLightID);
          }

          return cTerminator;
}

#line 162 "qHair.sufx"


#line 193 "Effects/D3D11/iLighting.shl"

float fLightScale;

float3 AccumulateHairLight(float3 vTangent, float3 vPositionWS, float3 vViewWS, HairShadeParams params, float fExtraAmbientScale = 1)
{

    // Light update system only counts lights that affect this object.
    int lightCount = nNumLights;

    float3 color = float3(0, 0, 0);
    float3 V = normalize(vViewWS);
    float3 T = normalize(vTangent);

    for (int i = 0; i < lightCount; ++i)
    {
        // volume lights will be treated as omni-directional.
        if (SuIsVolume(i))
        {
            color += SuComputeVolumeAmbient(vPositionWS, i);
        }
        else
        {
            float3 vLightDirWS, cLightColor;
            SuGetLightLocal(vPositionWS, i, vLightDirWS, cLightColor);

            int nLightID = nLightIndex[i]; // global ID used for shadows.

            // yes, I know this seems weird - but it's basically a way to control
            // spec vs diffuse through the light.
            float lightEmitsDiffuse = vLightParams[i].z;
            float lightEmitsSpecular = vLightParams[i].w;

            // if we get zero, we can save the BRDF eval, which is costly.
            float fShadowTerm = ComputeLightShadow(nLightID, vPositionWS);

            if (fShadowTerm > 0.00001)
            {
                float3 L = normalize(vLightDirWS);

                float3  reflection = TressFX_ComputeDiffuseSpecFactors(V, L, T);

                float3 cReflectedLight = reflection.x * cLightColor * lightEmitsDiffuse * params.cColor;
                cReflectedLight += reflection.y * cLightColor * lightEmitsSpecular;
                cReflectedLight += reflection.z * cLightColor * lightEmitsSpecular * params.cColor;
                cReflectedLight *= fShadowTerm * fLightScale;// * 0.16;

                color += max(float3(0, 0, 0), cReflectedLight);

            }
        }
    }

    return color;

}



#line 163 "qHair.sufx"


////////////////////////////////////////////////////////////////
// Code to Unroll K-buffer


float4 ComputeSushiRGBA(float2 pixelCoord, float depth, float4 vTangentCoverage, float3 baseColor)
{
    float3 vTangent = 2.0 * vTangentCoverage.xyz - 1.0;
    float3 vNDC = ScreenToNDC(float3(pixelCoord, depth), g_vViewport);
    float3 vPositionWS = NDCToWorld(vNDC, g_mInvViewProj);
    float3 vViewWS = g_vEye - vPositionWS;

    // TODO remove params, since we are using globals anyways.
    HairShadeParams params;

    params.cColor = baseColor;
    params.fRadius = g_FiberRadius;
    params.fSpacing = g_FiberSpacing;
    params.fAlpha = g_HairShadowAlpha;

    float3 color = AccumulateHairLight(vTangent, vPositionWS, vViewWS, params);
    return float4(color, vTangentCoverage.w);
}


#define HEAD_SHADING ComputeSushiRGBA
#define TAIL_SHADING ComputeSushiRGBA


#line 108 "../../amd_tressfx/src/shaders/TressFXPPLL.hlsl"



#ifndef KBUFFER_SIZE
#define KBUFFER_SIZE 8
#endif


#ifndef TAIL_SHADING

float4 TressFXTailColor(float2 pixelCoord, float depth, float4 vTangentCoverage, float3 baseColor)
{

    return float4(baseColor, vTangentCoverage.w);
}
#define TAIL_SHADING TressFXTailColor

#endif


#ifndef HEAD_SHADING

float4 TressFXHeadColor(float2 pixelCoord, float depth, float4 vTangentCoverage, float3 baseColor)
{
    return float4(baseColor, vTangentCoverage.w);

}

#define HEAD_SHADING TressFXHeadColor

#endif


#define MAX_FRAGMENTS 512
#define TAIL_COMPRESS 0



#if (KBUFFER_SIZE <= 16)
#define ALU_INDEXING			// avoids using an indexed array for better performance
#endif



#ifdef ALU_INDEXING
//--------------------------------------------------------------------------------------
// 
// Helper functions for storing the k-buffer into non-indexed registers for better 
// performance. For this code to work, KBUFFER_SIZE must be <= 16.
//
//--------------------------------------------------------------------------------------

uint GetUintFromIndex_Size16(uint4 V03, uint4 V47, uint4 V811, uint4 V1215, uint uIndex)
{
    uint u;
    u = uIndex == 1 ? V03.y : V03.x;
    u = uIndex == 2 ? V03.z : u;
    u = uIndex == 3 ? V03.w : u;
    u = uIndex == 4 ? V47.x : u;
    u = uIndex == 5 ? V47.y : u;
    u = uIndex == 6 ? V47.z : u;
    u = uIndex == 7 ? V47.w : u;
    u = uIndex == 8 ? V811.x : u;
    u = uIndex == 9 ? V811.y : u;
    u = uIndex == 10 ? V811.z : u;
    u = uIndex == 11 ? V811.w : u;
    u = uIndex == 12 ? V1215.x : u;
    u = uIndex == 13 ? V1215.y : u;
    u = uIndex == 14 ? V1215.z : u;
    u = uIndex == 15 ? V1215.w : u;
    return u;
}

void StoreUintAtIndex_Size16(inout uint4 V03, inout uint4 V47, inout uint4 V811, inout uint4 V1215, uint uIndex, uint uValue)
{
    V03.x = (uIndex == 0) ? uValue : V03.x;
    V03.y = (uIndex == 1) ? uValue : V03.y;
    V03.z = (uIndex == 2) ? uValue : V03.z;
    V03.w = (uIndex == 3) ? uValue : V03.w;
    V47.x = (uIndex == 4) ? uValue : V47.x;
    V47.y = (uIndex == 5) ? uValue : V47.y;
    V47.z = (uIndex == 6) ? uValue : V47.z;
    V47.w = (uIndex == 7) ? uValue : V47.w;
    V811.x = (uIndex == 8) ? uValue : V811.x;
    V811.y = (uIndex == 9) ? uValue : V811.y;
    V811.z = (uIndex == 10) ? uValue : V811.z;
    V811.w = (uIndex == 11) ? uValue : V811.w;
    V1215.x = (uIndex == 12) ? uValue : V1215.x;
    V1215.y = (uIndex == 13) ? uValue : V1215.y;
    V1215.z = (uIndex == 14) ? uValue : V1215.z;
    V1215.w = (uIndex == 15) ? uValue : V1215.w;
}


#define GET_DEPTH_AT_INDEX(uIndex) GetUintFromIndex_Size16(kBufferDepthV03, kBufferDepthV47, kBufferDepthV811, kBufferDepthV1215, uIndex)
#define GET_DATA_AT_INDEX( uIndex) GetUintFromIndex_Size16(kBufferDataV03, kBufferDataV47, kBufferDataV811, kBufferDataV1215, uIndex)
#define GET_COLOR_AT_INDEX( uIndex) GetUintFromIndex_Size16(kBufferStrandColorV03, kBufferStrandColorV47, kBufferStrandColorV811, kBufferStrandColorV1215, uIndex)
#define STORE_DEPTH_AT_INDEX(uIndex, uValue) StoreUintAtIndex_Size16(kBufferDepthV03, kBufferDepthV47, kBufferDepthV811, kBufferDepthV1215, uIndex, uValue)
#define STORE_DATA_AT_INDEX(uIndex, uValue) StoreUintAtIndex_Size16(kBufferDataV03, kBufferDataV47, kBufferDataV811, kBufferDataV1215, uIndex, uValue)
#define STORE_COLOR_AT_INDEX(uIndex, uValue) StoreUintAtIndex_Size16(kBufferStrandColorV03, kBufferStrandColorV47, kBufferStrandColorV811, kBufferStrandColorV1215, uIndex, uValue)

#else

//#define GET_DEPTH_AT_INDEX(uIndex) kBuffer[uIndex].depth
//#define GET_DATA_AT_INDEX(uIndex) kBuffer[uIndex].data
//#define STORE_DEPTH_AT_INDEX(uIndex, uValue) kBuffer[uIndex].depth = uValue
//#define STORE_DATA_AT_INDEX( uIndex, uValue) kBuffer[uIndex].data = uValue
#define GET_DEPTH_AT_INDEX(uIndex) kBuffer[uIndex].x
#define GET_DATA_AT_INDEX(uIndex) kBuffer[uIndex].y
#define STORE_DEPTH_AT_INDEX(uIndex, uValue) kBuffer[uIndex].x = uValue
#define STORE_DATA_AT_INDEX( uIndex, uValue) kBuffer[uIndex].y = uValue

#endif


#define NODE_DATA(x) LinkedListSRV[x].data
#define NODE_NEXT(x) LinkedListSRV[x].uNext
#define NODE_DEPTH(x) LinkedListSRV[x].depth
#define NODE_COLOR(x) LinkedListSRV[x].color



Texture2D<int>    tFragmentListHead;
StructuredBuffer<PPLL_STRUCT> LinkedListSRV;



float4 UnpackUintIntoFloat4(uint uValue)
{
    return float4(((uValue & 0xFF000000) >> 24) / 255.0, ((uValue & 0x00FF0000) >> 16) / 255.0, ((uValue & 0x0000FF00) >> 8) / 255.0, ((uValue & 0x000000FF)) / 255.0);
}

float4 ExtractHairColor(float2 pixelCoord, float depth, float4 data)
{
    return data;
}

float4 ExtractHairColor(float2 pixelCoord, float depth, float4 data, float4 color)
{
    return data;
}





#define KBUFFER_TYPE uint2;





float4 GatherLinkedList(float2 vfScreenAddress)
{
    uint2 vScreenAddress = uint2(vfScreenAddress);
    uint pointer = tFragmentListHead[vScreenAddress];

    float4 outColor = float4(0, 0, 0, 1);


    if (pointer == FRAGMENT_LIST_NULL)
        discard;

    ASSERT(pointer >= 0 && pointer < FRAGMENT_LIST_NULL)


#ifdef ALU_INDEXING

        uint4 kBufferDepthV03, kBufferDepthV47, kBufferDepthV811, kBufferDepthV1215;
    uint4 kBufferDataV03, kBufferDataV47, kBufferDataV811, kBufferDataV1215;
    uint4 kBufferStrandColorV03, kBufferStrandColorV47, kBufferStrandColorV811, kBufferStrandColorV1215;
    kBufferDepthV03 = uint4(asuint(100000.0f), asuint(100000.0f), asuint(100000.0f), asuint(100000.0f));
    kBufferDepthV47 = uint4(asuint(100000.0f), asuint(100000.0f), asuint(100000.0f), asuint(100000.0f));
    kBufferDepthV811 = uint4(asuint(100000.0f), asuint(100000.0f), asuint(100000.0f), asuint(100000.0f));
    kBufferDepthV1215 = uint4(asuint(100000.0f), asuint(100000.0f), asuint(100000.0f), asuint(100000.0f));
    kBufferDataV03 = uint4(0, 0, 0, 0);
    kBufferDataV47 = uint4(0, 0, 0, 0);
    kBufferDataV811 = uint4(0, 0, 0, 0);
    kBufferDataV1215 = uint4(0, 0, 0, 0);
    kBufferStrandColorV03 = uint4(0, 0, 0, 0);
    kBufferStrandColorV47 = uint4(0, 0, 0, 0);
    kBufferStrandColorV811 = uint4(0, 0, 0, 0);
    kBufferStrandColorV1215 = uint4(0, 0, 0, 0);

#else

        KBUFFER_TYPE kBuffer[KBUFFER_SIZE];

    [unroll]
    for (int t = 0; t < KBUFFER_SIZE; ++t)
    {
        //kBuffer[t].y = 0;
        //kBuffer[t].x = asuint(100000.0);
        STORE_DEPTH_AT_INDEX(t, asuint(100000.0));
        STORE_DATA_AT_INDEX(t, 0);
        compile error
    }
#endif

    // Get first K elements.
    for (int p = 0; p < KBUFFER_SIZE; ++p)
    {
        if (pointer != FRAGMENT_LIST_NULL)
        {
            STORE_DEPTH_AT_INDEX(p, NODE_DEPTH(pointer));
            STORE_DATA_AT_INDEX(p, NODE_DATA(pointer));
            STORE_COLOR_AT_INDEX(p, NODE_COLOR(pointer));
            pointer = NODE_NEXT(pointer);
        }
    }

    float4 fcolor = float4(0, 0, 0, 1);

    float3 tailColor;
    [allow_uav_condition]
    for (int iFragment = 0; iFragment < MAX_FRAGMENTS && pointer != FRAGMENT_LIST_NULL; ++iFragment)
    {
        if (pointer == FRAGMENT_LIST_NULL) break;

        int id = 0;
        float max_depth = 0;

        // find the furthest node in array
        for (int i = 0; i < KBUFFER_SIZE; i++)
        {
            float fDepth = asfloat(GET_DEPTH_AT_INDEX(i));
            if (max_depth < fDepth)
            {
                max_depth = fDepth;
                id = i;
            }
        }


        uint data = NODE_DATA(pointer);
        uint color = NODE_COLOR(pointer);
        uint nodeDepth = NODE_DEPTH(pointer);
        float fNodeDepth = asfloat(nodeDepth);


        // If the node in the linked list is nearer than the furthest one in the local array, exchange the node 
        // in the local array for the one in the linked list.
        if (max_depth > fNodeDepth)
        {
            uint tmp = GET_DEPTH_AT_INDEX(id);
            STORE_DEPTH_AT_INDEX(id, nodeDepth);
            fNodeDepth = asfloat(tmp);

            tmp = GET_DATA_AT_INDEX(id);
            STORE_DATA_AT_INDEX(id, data);
            data = tmp;

            tmp = GET_COLOR_AT_INDEX(id);
            STORE_COLOR_AT_INDEX(id, color);
            color = tmp;

        }

        float4 vData = UnpackUintIntoFloat4(data);
#if TAIL_COMPRESS
        float4 vColor = UnpackUintIntoFloat4(color);
        fcolor.w = mad(-fcolor.w, vColor.w, fcolor.w);
#else
        float4 vColor = UnpackUintIntoFloat4(color);
        float4 fragmentColor = TAIL_SHADING(vfScreenAddress, fNodeDepth, vData, vColor);
        //fragmentColor = float4( max(float(iFragment)/255.0,255.0)/255.0, iFragment <= 255 ? float(iFragment%255) : 0, 0, 1); 
        fcolor.xyz = mad(-fcolor.xyz, fragmentColor.w, fcolor.xyz) + fragmentColor.xyz * fragmentColor.w;
        fcolor.w = mad(-fcolor.w, fragmentColor.w, fcolor.w);
#endif

        pointer = NODE_NEXT(pointer);
    }
#if TAIL_COMPRESS
    float fTailAlphaInv = fcolor.w;
    fcolor.xyzw = float4(0, 0, 0, 1);
#endif

    // Blend the k nearest layers of fragments from back to front, where k = MAX_TOP_LAYERS_EYE
    for (int j = 0; j < KBUFFER_SIZE; j++)
    {
        int id = 0;
        float max_depth = 0;


        // find the furthest node in the array
        for (int i = 0; i < KBUFFER_SIZE; i++)
        {
            float fDepth = asfloat(GET_DEPTH_AT_INDEX(i));
            if (max_depth < fDepth)
            {
                max_depth = fDepth;
                id = i;
            }
        }

        // take this node out of the next search
        uint nodeDepth = GET_DEPTH_AT_INDEX(id);
        uint data = GET_DATA_AT_INDEX(id);
        uint color = GET_COLOR_AT_INDEX(id);

        // take this node out of the next search
        STORE_DEPTH_AT_INDEX(id, 0);

        // Use high quality shading for the nearest k fragments
        float fDepth = asfloat(nodeDepth);
        float4 vData = UnpackUintIntoFloat4(data);
        float4 vColor = UnpackUintIntoFloat4(color);
        float4 fragmentColor = HEAD_SHADING(vfScreenAddress, fDepth, vData, vColor);
#if TAIL_COMPRESS
        fragmentColor.w = 1 - (1 - fragmentColor.w) * fTailAlphaInv;
        //fTailAlphaInv = 1;
#endif
        // Blend the fragment color
        fcolor.xyz = mad(-fcolor.xyz, fragmentColor.w, fcolor.xyz) + fragmentColor.xyz * fragmentColor.w;
        fcolor.w = fcolor.w * (1 - fragmentColor.w);//mad(-fcolor.w, fragmentColor.w, fcolor.w);
    }
    outColor = fcolor;
    return outColor;
}

#line 192 "qHair.sufx"
///////////////////////////////////////////////////////////////

struct VS_OUTPUT_SCREENQUAD
{
    float4 vPosition : SV_POSITION;
    float2 vTex      : TEXCOORD;
};

static const float2 Positions[] = { {-1, -1}, {1, -1}, {-1,1}, {1,1} };

VS_OUTPUT_SCREENQUAD main_vs(uint gl_VertexID : SV_VertexID)
{
    VS_OUTPUT_SCREENQUAD output;
    //gl_Position = vec4( vPosition.xy, 0.5, 1);
    output.vPosition = float4(Positions[gl_VertexID].xy, 0, 1);

    // y down.
    output.vTex = float2(Positions[gl_VertexID].x, -Positions[gl_VertexID].y) * 0.5 + 0.5;
    return output;
}

[earlydepthstencil]
float4 main_ps(VS_OUTPUT_SCREENQUAD input) : SV_Target
{
    return GatherLinkedList(input.vPosition.xy);
}


technique11 TressFX2
{
	pass P0
	{
        SetVertexShader(CompileShader(vs_5_0, main_vs()));
        SetPixelShader(CompileShader(ps_5_0, main_ps()));
	}
}