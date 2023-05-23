float3 ScreenToNDC(float3 vScreenPos, float4 viewport)
{
	float2 xy = vScreenPos.xy;

	// add viewport offset.
	xy += viewport.xy;

	// scale by viewport to put in 0 to 1
	xy /= viewport.zw;

	// shift and scale to put in -1 to 1. y is also being flipped.
	xy.x = (2*xy.x) - 1;
	xy.y = 1 - (2*xy.y);

	return float3(xy, vScreenPos.z);

}

float3 NDCToWorld(float3 vNDC, float4x4 mInvViewProj)
{
	float4 pos = mul(mInvViewProj, float4(vNDC, 1) );
	
	return pos.xyz/pos.w;
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
	xy.x = (2*xy.x) + 1;
	xy.y = 1 - (2*xy.y);

	float4 pos = mul(invViewProj, float4(xy.x, xy.y, vScreenPos.z, 1) );
	//float4 pos = float4(xy.x, xy.y, 1, 1);
	//pos *= sv_pos.w;
	
	return pos.xyz/pos.w;
}