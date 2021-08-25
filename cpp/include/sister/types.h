#include "stdint.h"
// intrinsics
#include <smmintrin.h>
#include <emmintrin.h>
// memset
#include "string.h"
#include "assert.h"

typedef float float32;
typedef double float64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int8_t sint8;
typedef int16_t sint16;
typedef int32_t sint32;

#ifdef WIN32
    #define FORCEINLINE __forceinline
    #define ALIGN32 __declspec(align(32))
#else
    #define FORCEINLINE inline __attribute__((always_inline))
    #define ALIGN32 __attribute__ ((aligned(32)))
#endif

// saturate cast (uint8)
template<typename _Tp> static inline _Tp saturate_cast(uint8 v) { return _Tp(v); }

// access functions (DSI)
inline uint16* getDSIcell(uint16* dsi, sint32 width, sint32 disp, sint32 i, sint32 j, sint32 k)
{
    return dsi + i*(disp*width) + j*disp + k;
}

// access functions (RGB image)
inline uint8* getImagePixel(uint8* base, uint32 width, int j, int i)
{
	return base+i*width+j;
}

#pragma once
