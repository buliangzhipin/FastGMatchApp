#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <thread>
#include <vector>
#include "kernel.cuh"
#include "utilities.h"



class MainProcessGPU {

public:
	MainProcessGPU(int nx, int ny, int scaleNum, int scaleMaxNum, float scaleRatio);
	~MainProcessGPU();
	void loadFeature(const char* workdir);
	void threadFunction(float scale, int KTime);
	void calPoint(float *inImg, int *out);
	void savePoint(float *inImg);
	void saveFeatures(const char *fileName);
	
	

/*base information of the image*/
private:
	float *inImg;
	int xsize;
	int ysize;
	float scaleG;
	float scaleMaxG;
	float scaleRatio;
	int KMaxTimes;
	int transType = 0;

/*Gpu memory allocation*/
private:
	WorkIIRGPU *workiir;
	WorkIIRGPU *workiir2[(2 * PANGLE + 1) * 2];
	cudaStream_t pStream[(2 * PANGLE + 1) * 2];

/**/
private:
	float *d_dirHist;
	float *diffXImgGPU;
	float *diffYImgGPU;
	float **histFourS1GPU;
	float **histFourS4GPU;

/*calculate the features and match*/
private:
	std::vector<std::thread> threadPool;
	std::vector<Features> tmplList;
	Features **features;
	Features *featuresTmpl;
	WorkMT  *workMT;

private:
	char tmplDFileName[MAX_FILENAME];  /* template data file name */
	


};