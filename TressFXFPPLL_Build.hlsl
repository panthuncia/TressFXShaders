RWTexture2D<int>    tRWFragmentListHead;

RWStructuredBuffer<PPLL_STRUCT> LinkedListUAV;

int nNodePoolSize;

uint PackFloat4IntoUint(float4 vValue)
{
    return (((uint)(vValue.x * 255)) << 24) | (((uint)(vValue.y * 255)) << 16) | (((uint)(vValue.z * 255)) << 8) | (uint)(vValue.w * 255);
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