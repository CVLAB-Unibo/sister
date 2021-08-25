#include "stereoalgo.h"

inline void swapPointers(uint16*& p1, uint16*& p2)
{
    uint16* temp = p1;
    p1 = p2;
    p2 = temp;
}

inline sint32 adaptP2(const float32& alpha, const uint16& I_p, const uint16& I_pr, const int& gamma, const int& P2min)
{
    sint32 result;
    result = (sint32)(-alpha * abs((sint32)I_p-(sint32)I_pr)+gamma);
    if (result < P2min)
        result = P2min;
    return result;
}

// accumulate along paths
// variable P2 param
void accumulateCostsSSE(uint16* &dsi, uint8* im1_data, uint16* &sum, int height, int width, int dispCount, int paramP1, int paramP2min, int NPaths)
{
	/* Params */
	const uint16 paramInvalidDispCost = 0;//m_params.InvalidDispCost; 
	const int paramNoPasses = 2;//m_params.NoPasses;
	const uint16 MAX_SGM_COST = UINT16_MAX;
   
	// change params for fixed, if necessary
	const float32 paramAlpha = 0.f; //m_params.Alpha;
	const sint32 paramGamma = 100.; //m_params.Gamma;

	const int width2 = width+2;
	const int maxDisp = dispCount - 1;
	const int disp = maxDisp+1;
	const int dispP2 = disp+8;

	// accumulated cost along path r
	// two extra elements for -1 and maxDisp+1 disparity
	// current and last line (or better, two buffers)
	uint16* L_r0	  = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
	uint16* L_r0_last = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
	uint16* L_r1	  = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;
	uint16* L_r1_last = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;
	uint16* L_r2_last = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
	uint16* L_r3_last = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;
	
	/* image line pointers */
	uint8* im1_data_line_last = NULL;
	uint8* im1_data_line = NULL;

	/* left border */
	memset(&L_r1[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
	memset(&L_r1_last[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
	L_r1[-dispP2 - 1] = MAX_SGM_COST;
	L_r1_last[-dispP2 - 1] = MAX_SGM_COST;
	memset(&L_r3_last[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
	L_r3_last[-dispP2 - 1] = MAX_SGM_COST;

	/* right border */
	memset(&L_r1[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));
	memset(&L_r1_last[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));
	memset(&L_r3_last[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));

	// min L_r cache
	uint16 minL_r0_Array[2];
	uint16* minL_r0 = &minL_r0_Array[0];
	uint16* minL_r0_last = &minL_r0_Array[1]; 
	uint16* minL_r1 = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;
	uint16* minL_r1_last = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;
	uint16* minL_r2_last = (uint16*) _mm_malloc(width*sizeof(uint16),16);
	uint16* minL_r3_last = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;

	minL_r1[-1] =  minL_r1_last[-1] = 0;
	minL_r1[width] = minL_r1_last[width] = 0;
	minL_r3_last[-1] = 0;
	minL_r3_last[width] = 0;
	
	// border cases L_r0[0 - disp], L_r1,2,3 is maybe not needed, as done above
	L_r0_last[-1] = L_r1_last[-1] = L_r2_last[-1] = L_r3_last[-1] = MAX_SGM_COST;
	L_r0_last[disp] = L_r1_last[disp] = L_r2_last[disp] = L_r3_last[disp] = MAX_SGM_COST;
	L_r0[-1] = L_r1[-1] = MAX_SGM_COST;
	L_r0[disp] = L_r1[disp] = MAX_SGM_COST;

	for (int pass = 0; pass < paramNoPasses; pass++) {
		int i1; int i2; int di;
		int j1; int j2; int dj;
		if (pass == 0) {
			/* top-down pass */
			i1 = 0; i2 = height; di = 1;
			j1 = 0; j2 = width;  dj = 1;
		} else {
			/* bottom-up pass */
			i1 = height-1; i2 = -1; di = -1;
			j1 = width-1; j2 = -1;  dj = -1;
		}
		im1_data_line = im1_data+i1*width;

		/* first line is simply costs C, except for path L_r0 */
		// first pixel
		uint16 minCost = MAX_SGM_COST;
		if (pass == 0) {
			for (int d=0; d < disp; d++) {
				uint16 cost = *getDSIcell(dsi, width, disp, i1, j1, d);
				if (cost == 255)
					cost = paramInvalidDispCost;
				L_r0_last[d] = cost;
				L_r1_last[j1*dispP2+d] = cost;
				L_r2_last[j1*dispP2+d] = cost;
				L_r3_last[j1*dispP2+d] = cost;
				if (cost < minCost) {
					minCost = cost;
				}
				*getDSIcell(sum, width, disp, i1,j1, d) = cost;
			}
		} else {
			for (int d=0; d < disp; d++) {
				uint16 cost = *getDSIcell(dsi, width, disp, i1, j1, d);
				if (cost == 255)
					cost = paramInvalidDispCost;
				L_r0_last[d] = cost;
				L_r1_last[j1*dispP2+d] = cost;
				L_r2_last[j1*dispP2+d] = cost;
				L_r3_last[j1*dispP2+d] = cost;
				if (cost < minCost) {
					minCost = cost;
				}
				*getDSIcell(sum, width, disp, i1,j1, d) += cost;
			}
		}
		*minL_r0_last = minCost;
		minL_r1_last[j1] = minCost;
		minL_r2_last[j1] = minCost;
		minL_r3_last[j1] = minCost;

		// rest of first line
		for (int j=j1+dj; j != j2; j += dj) {
			uint16 minCost = MAX_SGM_COST;
			*minL_r0 = MAX_SGM_COST;
			for (int d=0; d < disp; d++) {
				uint16 cost = *getDSIcell(dsi, width, disp, i1, j, d);
				if (cost == 255)
					cost = paramInvalidDispCost;
				if (NPaths != 0) {
					L_r1_last[j*dispP2+d] = cost;
					L_r2_last[j*dispP2+d] = cost;
					L_r3_last[j*dispP2+d] = cost;
				}

				if (cost < minCost ) {
					minCost = cost;
				}

				// minimum along L_r0
				sint32 minPropCost = L_r0_last[d]; // same disparity cost
				// P1 costs
				sint32 costP1m = L_r0_last[d-1]+paramP1;
				if (minPropCost > costP1m)
					minPropCost = costP1m;
				sint32 costP1p = L_r0_last[d+1]+paramP1;
				if (minPropCost > costP1p) {
					minPropCost =costP1p;
				}
				// P2 costs
				sint32 minCostP2 = *minL_r0_last;
				sint32 varP2 = adaptP2(paramAlpha, im1_data_line[j],im1_data_line[j-dj],paramGamma, paramP2min);
				if (minPropCost > minCostP2+varP2)
					minPropCost = minCostP2+varP2;
				// add offset
				minPropCost -= minCostP2;

				const uint16 newCost = saturate_cast<uint16>(cost + minPropCost);
				L_r0[d] = newCost;

				if (*minL_r0 > newCost) {
					*minL_r0 = newCost;
				}

				// cost sum
				if (pass == 0) {
					*getDSIcell(sum, width, disp, i1,j, d) = saturate_cast<uint16>(cost + minPropCost);
				} else {
					*getDSIcell(sum, width, disp, i1,j, d) += saturate_cast<uint16>(cost + minPropCost);
				}
				
			}
			if (NPaths != 0) {
				minL_r1_last[j] = minCost;
				minL_r2_last[j] = minCost;
				minL_r3_last[j] = minCost;
			}

			// swap L0 buffers
			swapPointers(L_r0, L_r0_last);
			swapPointers(minL_r0, minL_r0_last);

			// border cases: disparities -1 and disp
			L_r1_last[j*dispP2-1] = L_r2_last[j*dispP2-1] = L_r3_last[j*dispP2-1] = MAX_SGM_COST;
			L_r1_last[j*dispP2+disp] = L_r2_last[j*dispP2+disp] = L_r3_last[j*dispP2+disp] = MAX_SGM_COST;
			
			L_r1[j*dispP2-1] = MAX_SGM_COST;
			L_r1[j*dispP2+disp] = MAX_SGM_COST;
		}

		// same as im1_data_line in first iteration, because of boundaries!
		im1_data_line_last = im1_data+(i1+di)*width;

		// remaining lines
		for (int i=i1+di; i != i2; i+=di) {

			memset(L_r0_last, 0, sizeof(uint16)*disp);
			*minL_r0_last = 0;

			im1_data_line = im1_data+i*width;

			for (int j=j1; j != j2; j+=dj) {
				*minL_r0 = MAX_SGM_COST;
				__m128i minLr_08 = _mm_set1_epi16(MAX_SGM_COST); 
				__m128i minLr_18 = _mm_set1_epi16(MAX_SGM_COST); 
				__m128i minLr_28 = _mm_set1_epi16(MAX_SGM_COST); 
				__m128i minLr_38 = _mm_set1_epi16(MAX_SGM_COST); 

				const sint32 varP2_r0 = adaptP2(paramAlpha, im1_data_line[j], im1_data_line[j-dj],paramGamma, paramP2min);
				sint32 varP2_r1, varP2_r2, varP2_r3;
				if (NPaths != 0) {
					varP2_r1 = adaptP2(paramAlpha, im1_data_line[j], im1_data_line_last[j-dj],paramGamma, paramP2min);
					varP2_r2 = adaptP2(paramAlpha, im1_data_line[j], im1_data_line_last[j],paramGamma, paramP2min);
					varP2_r3 = adaptP2(paramAlpha, im1_data_line[j], im1_data_line_last[j+dj],paramGamma, paramP2min);
				}
				else 
				{
					varP2_r1 = 0;
					varP2_r2 = 0;
					varP2_r3 = 0;
				}

				//only once per point
				const __m128i varP2_r08 = _mm_set1_epi16((uint16) varP2_r0);
				const __m128i varP2_r18 = _mm_set1_epi16((uint16) varP2_r1);
				const __m128i varP2_r28 = _mm_set1_epi16((uint16) varP2_r2);
				const __m128i varP2_r38 = _mm_set1_epi16((uint16) varP2_r3);

				const __m128i minCostP28_r0 =  _mm_set1_epi16((uint16) (*minL_r0_last));
				const __m128i minCostP28_r1 =  _mm_set1_epi16((uint16) minL_r1_last[j-dj]);
				const __m128i minCostP28_r2 =  _mm_set1_epi16((uint16) minL_r2_last[j]);
				const __m128i minCostP28_r3 =  _mm_set1_epi16((uint16) minL_r3_last[j+dj]);

				const __m128i curP2cost8_r0 = _mm_adds_epu16(varP2_r08, minCostP28_r0);
				const __m128i curP2cost8_r1 = _mm_adds_epu16(varP2_r18, minCostP28_r1);
				const __m128i curP2cost8_r2 = _mm_adds_epu16(varP2_r28, minCostP28_r2);
				const __m128i curP2cost8_r3 = _mm_adds_epu16(varP2_r38, minCostP28_r3);

				int d=0;
				__m128i upper8_r0 = _mm_load_si128( (__m128i*)( L_r0_last+0-1 ) );
				int baseIndex_r2 = ((j)*dispP2)+d;

				const int baseIndex_r1 = ((j - dj)*dispP2) + d;
				__m128i upper8_r1 = _mm_load_si128((__m128i*)(L_r1_last + baseIndex_r1 - 1));
				__m128i upper8_r2 = _mm_load_si128( (__m128i*)( L_r2_last+baseIndex_r2-1 ) );
				const int baseIndex_r3 = ((j+dj)*dispP2)+d;
				__m128i upper8_r3 = _mm_load_si128( (__m128i*)( L_r3_last+baseIndex_r3-1 ) );
				const __m128i paramP18 = _mm_set1_epi16((uint16)paramP1);

				for (; d < disp-7; d+=8) {
//--------------------------------------------------------------------------------------------------------------------------------------------------------
					//to save sum of all paths
					__m128i newCost8_ges = _mm_setzero_si128();
					
					__m128i cost8;
					
					cost8 = _mm_load_si128( (__m128i*) getDSIcell(dsi, width, disp, i, j, d) );

//--------------------------------------------------------------------------------------------------------------------------------------------------------
					// minimum along L_r0
					if (NPaths == 0 || NPaths == 8 || NPaths == 16) {
						__m128i minPropCost8;
						
						const __m128i lower8_r0 = upper8_r0;
						upper8_r0 = _mm_load_si128( (__m128i*)( L_r0_last+d-1+8 ) );
						
						// P1 costs
						const __m128i costPm8_r0 = _mm_adds_epu16(lower8_r0, paramP18);

						const __m128i costPp8_r0 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r0, lower8_r0, 4), paramP18);

						minPropCost8 = _mm_alignr_epi8(upper8_r0, lower8_r0, 2);
						__m128i temp = _mm_min_epu16(costPp8_r0, costPm8_r0);
						minPropCost8 = _mm_min_epu16(minPropCost8, temp);
						minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r0);
						minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r0);


						const __m128i newCost8_r0 = _mm_adds_epu16(cost8, minPropCost8);

						_mm_storeu_si128((__m128i*) (L_r0_last+d) , newCost8_r0);

						//sum of all Paths
						newCost8_ges = newCost8_r0;

						minLr_08 = _mm_min_epu16(minLr_08, newCost8_r0);
					}
					if (NPaths != 0) {
//--------------------------------------------------------------------------------------------------------------------------------------------------------
						const int baseIndex_r1 = ((j-dj)*dispP2)+d;

						uint16* lastL = L_r1_last;
						uint16* L = L_r1;
						
						const __m128i lower8_r1 = upper8_r1;
						upper8_r1 = _mm_load_si128( (__m128i*)( lastL+baseIndex_r1-1+8 ) );
						const __m128i costPm8_r1 = _mm_adds_epu16(lower8_r1, paramP18);

						const __m128i costPp8_r1 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r1, lower8_r1, 4), paramP18);

						__m128i minPropCost8 = _mm_alignr_epi8(upper8_r1, lower8_r1, 2);
						__m128i temp = _mm_min_epu16(costPp8_r1, costPm8_r1);
						minPropCost8 = _mm_min_epu16(minPropCost8, temp);
						minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r1);
						minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r1);
						
						const __m128i newCost8_r1 = _mm_adds_epu16(cost8, minPropCost8);

						_mm_storeu_si128((__m128i*) (L+(j*dispP2)+d) , newCost8_r1);
						
						//sum of all Paths
						newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r1);

						minLr_18 = _mm_min_epu16(minLr_18, newCost8_r1);

//--------------------------------------------------------------------------------------------------------------------------------------------------------
						int baseIndex_r2 = ((j)*dispP2) + d;
						
						const __m128i lower8_r2 = upper8_r2;
						upper8_r2 = _mm_load_si128( (__m128i*)( L_r2_last+baseIndex_r2-1+8 ) );
						

						const __m128i costPm8_r2 = _mm_adds_epu16(lower8_r2, paramP18);

						const __m128i costPp8_r2 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r2, lower8_r2, 4), paramP18);

						minPropCost8 = _mm_alignr_epi8(upper8_r2, lower8_r2, 2);
						temp = _mm_min_epu16(costPp8_r2, costPm8_r2);
						minPropCost8 = _mm_min_epu16(temp, minPropCost8);
						minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r2);
						minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r2);
						
						const __m128i newCost8_r2 = _mm_adds_epu16(cost8, minPropCost8);

						_mm_storeu_si128((__m128i*) (L_r2_last+(j*dispP2)+d) , newCost8_r2);

						//sum of all Paths
						newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r2);

						minLr_28 = _mm_min_epu16(minLr_28,newCost8_r2);

//--------------------------------------------------------------------------------------------------------------------------------------------------------
						int baseIndex_r3 = ((j+dj)*dispP2)+d;
						
						const __m128i lower8_r3 = upper8_r3;
						upper8_r3 = _mm_load_si128( (__m128i*)( L_r3_last+baseIndex_r3-1+8 ) );

						const __m128i costPm8_r3 = _mm_adds_epu16(lower8_r3, paramP18);

						const __m128i costPp8_r3 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r3, lower8_r3, 4), paramP18);

						minPropCost8 = _mm_alignr_epi8(upper8_r3, lower8_r3, 2);
						minPropCost8 = _mm_min_epu16(minPropCost8, costPm8_r3);
						minPropCost8 = _mm_min_epu16(minPropCost8, costPp8_r3);
						minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r3);
						minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r3);

						const __m128i newCost8_r3 = _mm_adds_epu16(cost8, minPropCost8);

						_mm_storeu_si128((__m128i*) (L_r3_last+(j*dispP2)+d) , newCost8_r3);

						//sum of all Paths
						newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r3);

						minLr_38 = _mm_min_epu16(minLr_38, newCost8_r3);
//--------------------------------------------------------------------------------------------------------------------------------------------------------
					}
				   if (NPaths == 8) {
						if (pass == 0) {
							_mm_store_si128((__m128i*) getDSIcell(sum, width, disp, i,j, d) , newCost8_ges);
						} else {
							_mm_store_si128((__m128i*) getDSIcell(sum, width, disp, i,j, d) , _mm_adds_epu16(_mm_load_si128( (__m128i*) getDSIcell(sum, width, disp, i, j, d) ), newCost8_ges));
						}
					} else if ((NPaths == 0)) {
						if (pass == 0) {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   = L_r0_last[d+k];
						} else {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k) +=   L_r0_last[d+k];
						}
					} else if (NPaths == 1) {
						if (pass == 0) {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   = L_r1[j*dispP2+d+k];
						} else {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k) += L_r1[j*dispP2+d+k];
						}
					} else if (NPaths == 2) {
						if (pass == 0) {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   = L_r2_last[j*dispP2+d+k];
						} else {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   += L_r2_last[j*dispP2+d+k];
						}
					} else if (NPaths == 3) {
						if (pass == 0) {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   = L_r3_last[j*dispP2+d+k];
						} else {
							for (int k=0; k<8;k++)
								*getDSIcell(sum, width, disp, i,j, d+k)   += L_r3_last[j*dispP2+d+k];
						}
					}
				}
				*minL_r0_last = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_08),0);
				minL_r1[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_18), 0);
				minL_r2_last[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_28), 0);
				minL_r3_last[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_38), 0);
//--------------------------------------------------------------------------------------------------------------------------------------------------------
			}

			im1_data_line_last = im1_data_line;
			// exchange buffers - swap line buffers
			{
				// one-liners
				swapPointers(L_r1, L_r1_last);
				swapPointers(minL_r1, minL_r1_last);
			}
		}
	}
	 
	/* free all */
	_mm_free(L_r0-1);
	_mm_free(L_r0_last-1);
	_mm_free(L_r1-dispP2-1);
	_mm_free(L_r1_last-dispP2-1);
	_mm_free(L_r2_last-1);
	_mm_free(L_r3_last-dispP2-1);

	_mm_free(minL_r1-1);
	_mm_free(minL_r1_last-1);
	_mm_free(minL_r2_last);
	_mm_free(minL_r3_last-1);
}

void sgm(uint8* im1_data, int height, int width, int  dispCount, uint16* dsi, uint16* sum_dsi, int paramP1, int paramP2min, int Paths)
{
	accumulateCostsSSE(dsi, im1_data, sum_dsi, height, width, dispCount, paramP1, paramP2min, Paths);
}

void striped_sgm(uint8* im1_data, int height, int width, int  dispCount, uint16* dsi, uint16* sum_dsi, int paramP1, int paramP2min, int Paths, int numStrips, int numThreads)
{
	int border = 16;
#pragma omp parallel num_threads(numThreads)
	{
#pragma omp for schedule(static, numStrips/numThreads)
		for (int n = 0; n < numStrips; n++)
		{

		        sint32 startLine = n*(height / numStrips);
		        sint32 endLine = (n + 1)*(height / numStrips) - 1;

		        if (n == numStrips - 1)
			{
		            endLine = height - 1;
		        }

	                sint32 startLineWithBorder = (startLine-border>0?startLine-border:0);

	                int imgOffset = startLineWithBorder * width;
	                int dsiOffset = startLineWithBorder * width * (dispCount);

			sgm(im1_data, height, width, dispCount, dsi, sum_dsi, paramP1, paramP2min, Paths);
		}
	}
}


