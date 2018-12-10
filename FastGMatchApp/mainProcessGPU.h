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
	void calPoint(float *inImg, int *out);
	void savePoint(float *inImg,const char*fileName);
	void threadFunction(float scale, int KTime);
	void loadFeature(const char* workdir);

private:
	float *inImg;
	int xsize;
	int ysize;
	float scaleG;
	float scaleMaxG;
	float scaleRatio;
	int KMaxTimes;

	WorkIIRGPU *workiir;
	WorkIIRGPU *workiir2[(2 * PANGLE + 1) * 2];
	cudaStream_t pStream[(2 * PANGLE + 1) * 2];

	float *d_dirHist;
	float *diffXImgGPU;
	float *diffYImgGPU;
	float **histFourS1GPU;
	float **histFourS4GPU;
	std::vector<std::thread> threadPool;



	Features **features;
	Features *featuresTmpl;
	WorkMT  *workMT;

	//char fileName[MAX_FILENAME];
	//char dFileName[MAX_FILENAME];     /* Data file name*/
	char tmplDFileName[MAX_FILENAME];  /* template data file name */
	


};