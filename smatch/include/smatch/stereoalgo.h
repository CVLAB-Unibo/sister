#include "types.h"

/* AD-Census stereo algorithm */
void ad_census(uint8* im1_data, uint8* im2_data, int height, int width, int dispCount, uint16* dsi, sint32 numThreads);

/* SGM stereo algorithm */
void sgm(uint8* im1_data, int height, int width, int  dispCount, uint16* dsi, uint16* sum_dsi, int paramP1, int paramP2min, int Paths);

/* WTA */
void WTALeft_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);
void WTARight_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness);

/* Median Filter */
void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height);

/* LR Check */
void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold);

/* Confidence */
void disparity_agreement(float32* src, float32* agreement, int width, int height, int k);
