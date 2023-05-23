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