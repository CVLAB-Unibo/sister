/*
Credits to Ivan Krešo for rSGM implementation:
https://github.com/ivankreso/stereo-vision/tree/master/reconstruction/base/rSGM
*/

#include "stereoalgo.h"

// hamming distance between uint32
inline uint16 hamDist32(const uint32& x, const uint32& y)
{
    uint16 dist = 0, val = (uint16)(x ^ y);

    // Count the number of set bits
    while(val)
    {
        ++dist; 
        val &= val - 1;
    }

    return dist;
}

// access functions (Census image)
inline uint64* getCensusPixel(uint64* base, uint32 width, int j, int i)
{
	return base+i*width+j;
}

// single pixel test + sum update
inline uint64 censusTest(uint8* source, uint32 width, sint32 i, sint32 j,uint64 value, sint32 x, sint32 y)
{
	uint64 current = value * 2;
	uint8 result = *getImagePixel(source, width,j+x,i+y) - *getImagePixel(source, width,j-x,i-y)>0;
	return current + result;
}

// census transform
void censusTransform(uint8* source, uint64* dest, uint32 width, uint32 height, uint8 rw, uint8 rh)
{
	memset(dest, 0, width*height*sizeof(uint64));

	uint64 censusValue = 0;
	for (sint32 i=rh; i < (sint32)height-rh; i++)
		for (sint32 j=rw; j < (sint32)width-rw; j++)
		{
			for (int y = -rh; y <= rh; y++)
				for (int x = -rw; x <= rw; x++)
					censusValue = censusTest(source, width, i,j, censusValue,x,y);
			*getCensusPixel(dest,width,j,i) = (uint64)censusValue;
		}
}

// process Hamming costs per lines
void hammingCostLine(uint64* im1_data, uint64* im2_data, int width, int dispCount, uint16* dsi, int startLine, int endLine)
{
	// build LUT
	uint16 LUT[UINT16_MAX+1]; 
	for (int i=0; i < UINT16_MAX+1; i++)
	{
		LUT[i] = hamDist32(i,0);
	}

	for (int i=startLine;i < endLine;i++)
	{
		uint64* baseRow = im1_data+ width*i;
		uint64* matchRow = im2_data + width*i;

		for (int j=0;j < width;j++)
		{
			uint64* pBaseJ = baseRow + j;
			uint64* pMatchRowJmD = matchRow + j - dispCount +1;

			sint32 d = dispCount -1;
			for (; d >(sint32)j && d >= 0;d--)
			{
				*getDSIcell(dsi,width, dispCount, i,j,d) = 255;
				pMatchRowJmD++;
			}
			while (d >= 0)
			{
				uint64 index = *pBaseJ ^ *pMatchRowJmD;
				uint16 cost = (LUT[index&0xFFFF] + LUT[(index>>16) & 0xFFFF] + LUT[(index>>32) & 0xFFFF]) + LUT[index>>48];
				pMatchRowJmD++;
				*getDSIcell(dsi,width, dispCount, i,j,d) = cost;
				d--;
			}
		}
	}
}

// Process Hamming costs
void hammingCost(uint64* im1_data, uint64* im2_data, int height, int width, int dispCount, uint16* dsi, sint32 numThreads)
{
	// first 3 lines are empty
	for (int i=0;i<3;i++)
		for (int j=0; j < width; j++)
			for (int d=0; d <= dispCount-1;d++)
				*getDSIcell(dsi,width, dispCount, i,j,d) = 255;

	if (numThreads != 4) 
	{
#pragma omp parallel num_threads(2)
		{
#pragma omp sections nowait
			{
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data, width, dispCount, dsi, 2, height/2);
				}
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data,width, dispCount, dsi, height/2, height-2);
				}
			}
		}
	} else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
		{
#pragma omp sections nowait
			{
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data, width, dispCount, dsi, 2, height/4);
				}
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data,width, dispCount, dsi, height/4, height/2);
				}
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data,width, dispCount, dsi, height/2, height-height/4);
				}
#pragma omp section
				{
					hammingCostLine(im1_data, im2_data,width, dispCount, dsi, height-height/4, height-2);
				}
			}
		}
	}

	// last 3 lines are empty
	for (int i=0;i<3;i++)
		for (int j=0; j < width; j++)
			for (int d=0; d <= dispCount-1;d++)
				*getDSIcell(dsi,width, dispCount, i,j,d) = 255;
}

// AD-census algorithm
void ad_census(uint8* im1_data, uint8* im2_data, int height, int width, int dispCount, uint16* dsi, sint32 numThreads)
{
	uint64* im1_dataCensus = (uint64*)_mm_malloc(width*height*sizeof(uint64), 16);
	uint64* im2_dataCensus = (uint64*)_mm_malloc(width*height*sizeof(uint64), 16);

	censusTransform(im1_data, im1_dataCensus, width, height, 4, 3);
	censusTransform(im2_data, im2_dataCensus, width, height, 4, 3);

	hammingCost(im1_dataCensus, im2_dataCensus, height, width, dispCount, dsi, numThreads);
}

