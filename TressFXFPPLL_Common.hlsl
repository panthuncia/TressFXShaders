
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