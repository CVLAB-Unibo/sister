#include "stereoalgo.h"

inline void vecSortandSwap(__m128& a, __m128& b)
{
	__m128 temp = a;
	a = _mm_min_ps(a,b);
	b = _mm_max_ps(temp,b);
}

void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
{
	// check width restriction
	assert(width % 4 == 0);
	
	float32* destStart = dest;
	//  lines
	float32* line1 = source;
	float32* line2 = source + width;
	float32* line3 = source + 2*width;

	float32* end = source + width*height;

	dest += width;
	__m128 lastMedian = _mm_setzero_ps();

	do {
		// fill value
		const __m128 l1_reg = _mm_load_ps(line1);
		const __m128 l1_reg_next = _mm_load_ps(line1+4);
		__m128 v0 = l1_reg;
		__m128 v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 4));
		__m128 v2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 8));

		const __m128 l2_reg = _mm_load_ps(line2);
		const __m128 l2_reg_next = _mm_load_ps(line2+4);
		__m128 v3 = l2_reg;
		__m128 v4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 4));
		__m128 v5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 8));

		const __m128 l3_reg = _mm_load_ps(line3);
		const __m128 l3_reg_next = _mm_load_ps(line3+4);
		__m128 v6 = l3_reg;
		__m128 v7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 4));
		__m128 v8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 8));

		// find median through sorting network
		vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
		vecSortandSwap(v0, v1) ; vecSortandSwap(v3, v4) ; vecSortandSwap(v6, v7) ;
		vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
		vecSortandSwap(v0, v3) ; vecSortandSwap(v5, v8) ; vecSortandSwap(v4, v7) ;
		vecSortandSwap(v3, v6) ; vecSortandSwap(v1, v4) ; vecSortandSwap(v2, v5) ;
		vecSortandSwap(v4, v7) ; vecSortandSwap(v4, v2) ; vecSortandSwap(v6, v4) ;
		vecSortandSwap(v4, v2) ; 

		// comply to alignment restrictions
		const __m128i c = _mm_alignr_epi8(_mm_castps_si128(v4), _mm_castps_si128(lastMedian), 12);
		_mm_store_si128((__m128i*)dest, c);
		lastMedian = v4;

		dest+=4; line1+=4; line2+=4; line3+=4;

	} while (line3+4+4 <= end);

	memcpy(destStart, source, sizeof(float32)*(width+1));
	memcpy(destStart+width*height-width-1-3, source+width*height-width-1-3, sizeof(float32)*(width+1+3));
}


void WTALeft_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
	const uint32 factorUniq = (uint32)(1024*uniqueness);
	const sint32 disp = maxDisp+1;
	
	// find best by WTA
	float32* pDestDisp = dispImg;
	for (sint32 i=0;i < height; i++) {
		for (sint32 j=0;j < width; j++) {
			// WTA on disparity values
			
			uint16* pCost = getDSIcell(dsiAgg, width, disp, i,j,0);
			uint16* pCostBase = pCost;
			uint32 minCost = *pCost;
			uint32 secMinCost = minCost;
			int secBestDisp = 0;

			const uint32 end = (disp-1>j?j:disp-1);
			if (end == (uint32)disp-1) {
				uint32 bestDisp = 0;

				for (uint32 loop =0; loop < end;loop+= 8) {
					// load costs
					const __m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					const __m128i b = _mm_minpos_epu16(costs);
					const int minValue = _mm_extract_epi16(b,0);

				   if ((uint32)minValue < minCost) {
						minCost = (uint32)minValue;
						bestDisp = _mm_extract_epi16(b,1)+loop;
				   }
					pCost+=8;
				}

				// get value of second minimum
				pCost = pCostBase;
				pCost[bestDisp]=65535;

#ifdef USE_AVX2
				__m256i secMinVector = _mm256_set1_epi16(-1);
				const uint16* pCostEnd = pCost+disp;
				for (; pCost < pCostEnd;pCost+= 16) {
					// load costs
					__m256i costs = _mm256_load_si256((__m256i*)pCost);
					// get minimum for 8 values
					secMinVector = _mm256_min_epu16(secMinVector,costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector,0)),0);
				uint32 secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
				if (secMinCost2 < secMinCost)
					secMinCost = secMinCost2;
#else
				__m128i secMinVector = _mm_set1_epi16(-1);
				const uint16* pCostEnd = pCost+disp;
				for (; pCost < pCostEnd;pCost+= 8) {
					// load costs
					__m128i costs = _mm_load_si128((__m128i*)pCost);
					// get minimum for 8 values
					secMinVector = _mm_min_epu16(secMinVector,costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#endif
				pCostBase[bestDisp]=(uint16)minCost;
				
				// assign disparity
				if (1024*minCost <=  secMinCost*factorUniq) {
					*pDestDisp = (float)bestDisp;
				} else {
					bool check = false;
					if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) {
						check=true;
					} 
					if (bestDisp>0 && pCostBase[bestDisp-1] == secMinCost) {
						check=true;
					}
					if (!check) {
						*pDestDisp = -10;
					} else {
						*pDestDisp = (float)bestDisp;
					}
				}
				
			} else {
				int bestDisp = 0;
				// for start
				for (uint32 k=1; k <= end; k++) {
					pCost += 1;
					const uint16 cost = *pCost;
					if (cost < secMinCost) {
						if (cost < minCost) {
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						} else  {
							secMinCost = cost;
							secBestDisp = k;
						}
					}
				}
				// assign disparity
				if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
					*pDestDisp = (float)bestDisp;
				} else {
					*pDestDisp = -10;
				}
			}
			pDestDisp++;
		}
	}
}

void WTARight_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
	const uint32 factorUniq = (uint32)(1024*uniqueness); 

	const uint32 disp = maxDisp+1;
	//_ASSERT(disp <= 256);
	ALIGN32 uint16 store[256+32];
	store[15] = UINT16_MAX-1;
	store[disp+16] = UINT16_MAX-1;

	// find best by WTA
	float32* pDestDisp = dispImg;
	for (uint32 i=0;i < (uint32)height; i++)
	{
		for (uint32 j=0;j < (uint32)width;j++)
		{
			// WTA on disparity values
			int bestDisp = 0;
			uint16* pCost = getDSIcell(dsiAgg, width, disp, i,j,0);
			sint32 minCost = *pCost;
			sint32 secMinCost = minCost;
			int secBestDisp = 0;
			const uint32 maxCurrDisp = (disp-1<width-1-j?disp-1:width-1-j);

			if (maxCurrDisp == disp-1)
			{

				// transfer to linear storage, slightly unrolled
				for (uint32 k=0; k <= maxCurrDisp; k+=4)
				{
					store[k+16]=*pCost;
					store[k+16+1]=pCost[disp+1];
					store[k+16+2]=pCost[2*disp+2];
					store[k+16+3]=pCost[3*disp+3];
					pCost += 4*disp+4;
				}
				// search in there
				uint16* pStore = &store[16];
				const uint16* pStoreEnd = pStore+disp;
				for (; pStore < pStoreEnd; pStore+=8)
				{
					// load costs
					const __m128i costs = _mm_load_si128((__m128i*)pStore);
					// get minimum for 8 values
					const __m128i b = _mm_minpos_epu16(costs);
					const int minValue = _mm_extract_epi16(b,0);

					if (minValue < minCost)
					{
						minCost = minValue;
						bestDisp = _mm_extract_epi16(b,1)+(int)(pStore-&store[16]);
					}
					
				}

				// get value of second minimum
				pStore = &store[16];
				store[16+bestDisp]=65535;
#ifndef USE_AVX2
				__m128i secMinVector = _mm_set1_epi16(-1);
				for (; pStore < pStoreEnd;pStore+= 8)
				{
					// load costs
					__m128i costs = _mm_load_si128((__m128i*)pStore);
					// get minimum for 8 values
					secMinVector = _mm_min_epu16(secMinVector,costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#else
				__m256i secMinVector = _mm256_set1_epi16(-1);
				for (; pStore < pStoreEnd; pStore += 16)
				{
					// load costs
					__m256i costs = _mm256_load_si256((__m256i*)pStore);
					// get minimum for 8 values
					secMinVector = _mm256_min_epu16(secMinVector, costs);
				}
				secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
				int secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
				if (secMinCost2 < secMinCost)
					secMinCost = secMinCost2;
#endif

				// assign disparity
				if (1024U*minCost <=  secMinCost*factorUniq)
					*pDestDisp = (float)bestDisp;
				else
				{
					bool check = (store[16+bestDisp+1] == secMinCost);
					check = check  | (store[16+bestDisp-1] == secMinCost);
					if (!check)
						*pDestDisp = -10;
					else
						*pDestDisp = (float)bestDisp;
				}
				pDestDisp++;
			} 
			else 
			{
				// border case handling
				for (uint32 k=1; k <= maxCurrDisp; k++)
				{
					pCost += disp+1;
					const sint32 cost = (sint32)*pCost;
					if (cost < secMinCost)
						if (cost < minCost)
						{
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						}
						else
						{
							secMinCost = cost;
							secBestDisp = k;
						}
				}
				// assign disparity
				if (1024U*minCost <= factorUniq*secMinCost|| abs(bestDisp - secBestDisp) < 2  )
					*pDestDisp = (float)bestDisp;
				else
					*pDestDisp = -10;

				pDestDisp++;
			}
		}
	}
}


void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
	float* dispRow = dispImg;
	float* dispCheckRow = dispCheckImg;
	for (sint32 i=0;i < height;i++)
	{
		for (sint32 j=0;j < width;j++)
		{
			const float32 baseDisp = dispRow[j];
			if (baseDisp >= 0 && baseDisp <= j)
			{
				const float matchDisp = dispCheckRow[(int)(j-baseDisp)];

				sint32 diff = (sint32)(baseDisp - matchDisp);
				if (abs(diff) > lrThreshold)
					dispRow[j] = -10; // occluded or false match
			}
			else
				dispRow[j] = -10;
		}
		dispRow += width;
		dispCheckRow += width;
	}	
}

