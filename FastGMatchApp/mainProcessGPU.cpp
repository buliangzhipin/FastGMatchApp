#include "kernel.cuh"
#include "utilities.h"
#include "mainProcessGPU.h"

using namespace std;



	

	

	MainProcessGPU::MainProcessGPU(int nx, int ny, int scaleNum, int scaleMaxNum,float scaleRatio) {

		int nAngleCoef = 2 * PANGLE + 1;
		int nxy = nx * ny;
		this->scaleRatio = scaleRatio;

		//initial gpu mem
		xsize = nx;
		ysize = ny;
		inImg = (FPTYPE *)malloc(sizeof(FPTYPE) * nxy);
		workiir = (WorkIIRGPU *)malloc(sizeof(WorkIIRGPU));
		initialGPU4(workiir, nx, ny, 4,scaleMaxNum);
		for (int i = 0; i < (2 * PANGLE + 1) * 2; i++) {
			workiir2[i] = (WorkIIRGPU *)malloc(sizeof(WorkIIRGPU));
			if (i/ (2 * PANGLE + 1) == 0) {
				initialGPU2(workiir2[i], nx, ny, 2, (int)(scaleMaxNum *PI / SIGMA2 + ROUNDFRAC));
			}
			else {
				initialGPU2(workiir2[i], nx, ny, 2, scaleMaxNum);
			}
			
			cudaStreamCreate(&(pStream[i]));
		}

		int scaleTest = scaleNum;
		int KNumMax = 0;
		while (scaleTest <= scaleMaxNum) {
			scaleTest *= scaleRatio;
			KNumMax++;
		}
		KMaxTimes = KNumMax;


		cudaMalloc((void**)&d_dirHist, sizeof(FPTYPE) * nxy * nAngleCoef);

		cudaMallocHost((void**)&diffXImgGPU, sizeof(FPTYPE) * nxy);
		cudaMallocHost((void**)&diffYImgGPU, sizeof(FPTYPE) * nxy);

		histFourS1GPU = (float**)malloc(KNumMax * sizeof(float*));
		histFourS4GPU = (float**)malloc(KNumMax * sizeof(float*));

		for (int i = 0; i < KNumMax; i++)
		{
			cudaMallocHost((void**)histFourS1GPU + i, sizeof(FPTYPE) * nxy * nAngleCoef);
			cudaMallocHost((void**)histFourS4GPU + i, sizeof(FPTYPE) * nxy * nAngleCoef);
		}



		//initial features and featureTemplate mem
		features = nullptr;
		featuresTmpl = (Features *)malloc(sizeof(Features));
		


		featuresTmpl->relativePosL = (FPTYPE *)malloc(sizeof(FPTYPE) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
		featuresTmpl->relativePosScL = (int *)malloc(sizeof(int) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
		featuresTmpl->tmpVector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * 4 * 4);
		featuresTmpl->nx = nx;
		featuresTmpl->ny = ny;
		featuresTmpl->nxy = nxy;
		featuresTmpl->nAngleCoef = nAngleCoef;
		featuresTmpl->nEstFeature = 0;


		int  pos, posTheta, order, iTheta, ix, iy;
		double dTheta, cosTheta, sinTheta, x, y;

		//scaleG = scaleNum;
		//while (scaleG <= scaleMaxNum) {
		//	featuresTmpl->nEstFeature += NTEMPLATEROT * NTEMPLATEENLONG + 1;
		//	scaleG = (int)scaleG * TEMPLSCALERATIO;
		//}
		//featuresTmpl->featureL = (Feature *)malloc(sizeof(Feature) * featuresTmpl->nEstFeature);
		//for (int pos = 0; pos < featuresTmpl->nEstFeature; ++pos) {
		//	featuresTmpl->featureL[pos].vector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE);
		//}


		//int temp = 0;
		//scaleG = scaleNum;
		//while (scaleG <= scaleMaxNum) {
		//	features[temp] = (Features *)malloc(sizeof(Features));
		//	features[temp]->relativePosL = (FPTYPE *)malloc(sizeof(FPTYPE) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
		//	features[temp]->relativePosScL = (int *)malloc(sizeof(int) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
		//	features[temp]->tmpVector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * 4 * 4);
		//	features[temp]->nx = nx;
		//	features[temp]->ny = ny;
		//	features[temp]->nxy = nxy;
		//	features[temp]->nAngleCoef = nAngleCoef;
		//	features[temp]->nEstFeature = 0;
		//	features[temp]->nEstFeature += (int)(nxy / (scaleG * STEPRATIO * scaleG * STEPRATIO) + ROUNDFRAC);
		//	features[temp]->nEstFeature *= 2;
		//	features[temp]->featureL = (Feature *)malloc(sizeof(Feature) * features[temp]->nEstFeature);
		//	scaleG = (int)scaleG * SCALERATIO;
		//	for (pos = 0; pos < features[temp]->nEstFeature; ++pos) {
		//		features[temp]->featureL[pos].vector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE);
		//	}
		//	pos = 0;
		//	for (iTheta = 0; iTheta < NMAXTHETA; ++iTheta) {
		//		cosTheta = cos(2 * PI * iTheta / NMAXTHETA);
		//		sinTheta = sin(2 * PI * iTheta / NMAXTHETA);
		//		y = -(GRIDSIZE - 1.0) / GRIDSIZE;
		//		for (iy = 0; iy < GRIDSIZE; ++iy) {
		//			x = -(GRIDSIZE - 1.0) / GRIDSIZE;
		//			for (ix = 0; ix < GRIDSIZE; ++ix) {
		//				featuresTmpl->relativePosL[pos] = features[temp]->relativePosL[pos] = x * cosTheta - y * sinTheta; ++pos;// x
		//				featuresTmpl->relativePosL[pos] = features[temp]->relativePosL[pos] = x * sinTheta + y * cosTheta; ++pos;// y
		//				x = x + 2.0 / GRIDSIZE;
		//			}
		//			y = y + 2.0 / GRIDSIZE;
		//		}
		//	}
		//	temp++;
		//}






		/* Initialize workMT */
		workMT = (WorkMT *)malloc(sizeof(WorkMT));
		workMT->nx = nx;
		workMT->ny = ny;
		workMT->invFourTbl = (FPTYPE *)malloc(sizeof(FPTYPE) * 2 * NMAXTHETA * (2 * PANGLE + 1));
		workMT->rotTbl = (FPTYPE *)malloc(sizeof(FPTYPE) * 2 * NMAXTHETA * (2 * PANGLE));
		workMT->largeScaleRelTbl = (FPTYPE *)malloc(sizeof(FPTYPE) * 2 * NAPPROPOINT);
		workMT->largeScaleTbl = (int *)malloc(sizeof(int) * 2 * NAPPROPOINT);
		workMT->largeScaleATbl = (FPTYPE *)malloc(sizeof(FPTYPE) * NAPPROPOINT);
		dTheta = 2 * PI / NMAXTHETA;

		/* For inverse Fourier transform of max angle*/
		pos = 0;
		for (posTheta = 0; posTheta < NMAXTHETA; ++posTheta) {
			workMT->invFourTbl[pos++] = 0.5 / PI;
			for (order = 1; order <= PANGLE; ++order) {
				workMT->invFourTbl[pos++] = cos(dTheta * order * posTheta) / PI;
				workMT->invFourTbl[pos++] = -sin(dTheta * order * posTheta) / PI;
			}
		}

		/* For rotation of direction histogram */
		pos = 0;
		for (posTheta = 0; posTheta <= NMAXTHETA; ++posTheta) {
			workMT->rotTbl[pos++] = 1.0;
			for (order = 1; order <= PANGLE; ++order) {
				workMT->rotTbl[pos++] = cos(dTheta * order * posTheta);
				workMT->rotTbl[pos++] = sin(dTheta * order * posTheta);
			}
		}

		scaleG = scaleNum;
		scaleMaxG = scaleMaxNum;

		

	}

	void MainProcessGPU::loadFeature(const char* workdir) {


		int temp = 0, pos = 0, nx = xsize, ny = ysize, nAngleCoef = 2 * PANGLE + 1;
		int nxy = nx * ny;
		float scale = scaleG;
		int   posTheta, order, iTheta, ix, iy;
		double dTheta, cosTheta, sinTheta, x, y;
		features = (Features **)malloc(sizeof(Features*)*KMaxTimes);
		while (scale <= scaleMaxG) {
			features[temp] = (Features *)malloc(sizeof(Features));
			features[temp]->relativePosL = (FPTYPE *)malloc(sizeof(FPTYPE) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
			features[temp]->relativePosScL = (int *)malloc(sizeof(int) * NMAXTHETA * GRIDSIZE * GRIDSIZE * 2);
			features[temp]->tmpVector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * 4 * 4);
			features[temp]->nx = nx;
			features[temp]->ny = ny;
			features[temp]->nxy = nxy;
			features[temp]->nAngleCoef = nAngleCoef;
			features[temp]->nEstFeature = 0;
			features[temp]->nEstFeature += (int)(nxy / (scale * STEPRATIO * scale * STEPRATIO) + ROUNDFRAC);
			features[temp]->nEstFeature *= 2;
			features[temp]->featureL = (Feature *)malloc(sizeof(Feature) * features[temp]->nEstFeature);
			for (pos = 0; pos < features[temp]->nEstFeature; ++pos) {
				features[temp]->featureL[pos].vector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE);
			}
			pos = 0;
			for (iTheta = 0; iTheta < NMAXTHETA; ++iTheta) {
				cosTheta = cos(2 * PI * iTheta / NMAXTHETA);
				sinTheta = sin(2 * PI * iTheta / NMAXTHETA);
				y = -(GRIDSIZE - 1.0) / GRIDSIZE;
				for (iy = 0; iy < GRIDSIZE; ++iy) {
					x = -(GRIDSIZE - 1.0) / GRIDSIZE;
					for (ix = 0; ix < GRIDSIZE; ++ix) {
						featuresTmpl->relativePosL[pos] = features[temp]->relativePosL[pos] = x * cosTheta - y * sinTheta; ++pos;// x
						featuresTmpl->relativePosL[pos] = features[temp]->relativePosL[pos] = x * sinTheta + y * cosTheta; ++pos;// y
						x = x + 2.0 / GRIDSIZE;
					}
					y = y + 2.0 / GRIDSIZE;
				}
			}
			scale *= scaleRatio;
			temp++;
		}


		loadFeatures(workdir, featuresTmpl);
	}

	MainProcessGPU::~MainProcessGPU() {
		int scaleNum = scaleG;
		int scaleMaxNum = scaleMaxG;


		//free gpu
		free(inImg);
		deleteGPU4(workiir);
		free(workiir);
		for (int i = 0; i < (2 * PANGLE + 1) * 2; i++) {
			deleteGPU2(workiir2[i]);
			free(workiir2[i]);
			cudaStreamDestroy(pStream[i]);
		}
		//free(workiir2);

		cudaFree(d_dirHist);
		cudaFree(diffXImgGPU);
		cudaFree(diffYImgGPU);


		//free hist gpu
		for (int i = 0; i < KMaxTimes; i++)
		{
			cudaFree(histFourS1GPU[i]);
			cudaFree(histFourS4GPU[i]);
		}
		free(histFourS1GPU);
		free(histFourS4GPU);


		//free featuresTmpl
		for (int pos = 0; pos < featuresTmpl->nEstFeature; ++pos) {
			free(featuresTmpl->featureL[pos].vector);
		}
		free(featuresTmpl->featureL);
		free(featuresTmpl->relativePosL);
		free(featuresTmpl->relativePosScL);
		free(featuresTmpl->tmpVector);
		free(featuresTmpl);

		//free features
		int temp = 0;
		if (features !=nullptr) {
			while (scaleNum <= scaleMaxNum) {
				for (int pos = 0; pos < features[temp]->nEstFeature; ++pos) {
					free(features[temp]->featureL[pos].vector);
				}

				free(features[temp]->featureL);
				free(features[temp]->relativePosL);
				free(features[temp]->relativePosScL);
				free(features[temp]->tmpVector);
				free(features[temp]);

				scaleNum = (int)scaleNum * scaleRatio;
				temp++;
			}
			free(features);
		}

		//free workMT
		free(workMT->invFourTbl);
		free(workMT->rotTbl);
		free(workMT->largeScaleATbl);
		free(workMT->largeScaleRelTbl);
		free(workMT->largeScaleTbl);
		free(workMT);

		cudaDeviceReset();
	}

	void MainProcessGPU::calPoint(float *inImg, int *out) {

		int nxy = xsize * ysize;
		int K4, K2, K, pos, iFeature;

		float scale = scaleG;
		float scaleMax = scaleMaxG;



		for (int i = 0; i < KMaxTimes; i++) {
			features[i]->nFeature = 0;
		}

		clock_t start = clock();
		int kTimes = 0;

		CHECK(cudaMemcpy(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice));

		while (scale <= scaleMax) {
			/* Calculate gradient images */
			K4 = (int)(PI * scale * GRADRATIO / (GRIDSIZE * SIGMA4) + ROUNDFRAC);

			gaussDiffGPUAsync(diffXImgGPU, diffYImgGPU, inImg, workiir, xsize, ysize, K4, pStream);

			/* Fourier series expression of directions */
			calDirHistPointGPUAsync(workiir->d_diffImg, workiir->d_blurImg, d_dirHist, xsize, ysize);

			/* Blurred directions for large scale */
			K = (int)(PI * scale / SIGMA2 + ROUNDFRAC);
			K2 = (int)(PI * scale / (SIGMA2 * GRIDSIZE) + ROUNDFRAC);

			gaussSmoothGPUAsyncHeight(histFourS1GPU[kTimes], histFourS4GPU[kTimes], d_dirHist, workiir2, xsize, ysize, K, K2, pStream, 2 * PANGLE + 1);
			cudaDeviceSynchronize();
			threadPool.push_back(thread(&MainProcessGPU::threadFunction,this, scale, kTimes));
			scale *= scaleRatio;
			kTimes++;

		}/* End of transformation loop */
		for (auto &th : threadPool)
		{
			th.join();
		}
		threadPool.clear();


		int maxFeature = 0;
		for (int i = 0; i < KMaxTimes; i++) {
			maxFeature += features[i]->nFeature;
		}

		int iFeatureTmpl, iVec;

		int shift = 0;
		float recordMax = 0;
		int position = 0;

		for (int i = 0; i < KMaxTimes; i++) {
			pos = 0;
			for (iFeature = 0; iFeature < features[i]->nFeature; ++iFeature) {
				FPTYPE *vector = (features[i]->featureL)[iFeature].vector;
				for (iFeatureTmpl = 0; iFeatureTmpl < featuresTmpl->nFeature; ++iFeatureTmpl) {
					FPTYPE *vectorTmpl = (featuresTmpl->featureL)[iFeatureTmpl].vector;
					FPTYPE val = 0.0;
					for (iVec = 0; iVec < (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE; ++iVec) {
						val += vector[iVec] * vectorTmpl[iVec];
					}

					if (val > recordMax) {
						shift = i;
						recordMax = val;
						position = pos;
					}
					++pos;
				}
			}
		}

		iFeature = position / featuresTmpl->nFeature;

		Feature *featureTemp = &((features[shift]->featureL)[iFeature]);
		out[0] = featureTemp->ix;
		out[1] = featureTemp->iy;
		out[2] = featureTemp->scale;
		out[3] = (int)(recordMax * 100);
		out[4] = featureTemp->iTheta;
	}


	void MainProcessGPU::savePoint(float *inImg,const char *fileName) {

		featuresTmpl->nEstFeature = KMaxTimes;
		featuresTmpl->nFeature = 0;
		featuresTmpl->featureL = (Feature *)malloc(sizeof(Feature) * featuresTmpl->nEstFeature);
		for (int pos = 0; pos < featuresTmpl->nEstFeature; ++pos) {
			featuresTmpl->featureL[pos].vector = (FPTYPE *)malloc(sizeof(FPTYPE) * (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE);
		}
		
		
		for (int i = 0; i < 1; i++) {
			int position = 0;
			double cosTheta, sinTheta, x, y;
			for (int iTheta = 0; iTheta < NMAXTHETA; ++iTheta) {
				cosTheta = cos(2 * PI * iTheta / NMAXTHETA);
				sinTheta = sin(2 * PI * iTheta / NMAXTHETA);
				y = -(GRIDSIZE - 1.0) / GRIDSIZE;
				for (int iy = 0; iy < GRIDSIZE; ++iy) {
					x = -(GRIDSIZE - 1.0) / GRIDSIZE;
					for (int ix = 0; ix < GRIDSIZE; ++ix) {
						featuresTmpl->relativePosL[position] = x * cosTheta - y * sinTheta; ++position;// x
						featuresTmpl->relativePosL[position] = x * sinTheta + y * cosTheta; ++position;// y
						x = x + 2.0 / GRIDSIZE;
					}
					y = y + 2.0 / GRIDSIZE;
				}
			}
		}


		int nxy = xsize * ysize;
		int K4, K2, K, pos, iFeature = 0;

		float scale = scaleG;
		float scaleMax = scaleMaxG;


		clock_t start = clock();
		int kTimes = 0;

		CHECK(cudaMemcpy(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice));

		while (scale <= scaleMax) {
			/* Calculate gradient images */
			K4 = (int)(PI * scale * GRADRATIO / (GRIDSIZE * SIGMA4) + ROUNDFRAC);

			gaussDiffGPUAsync(diffXImgGPU, diffYImgGPU, inImg, workiir, xsize, ysize, K4, pStream);

			/* Fourier series expression of directions */
			calDirHistPointGPUAsync(workiir->d_diffImg, workiir->d_blurImg, d_dirHist, xsize, ysize);

			/* Blurred directions for large scale */
			K = (int)(PI * scale / SIGMA2 + ROUNDFRAC);
			K2 = (int)(PI * scale / (SIGMA2 * GRIDSIZE) + ROUNDFRAC);

			gaussSmoothGPUAsyncHeight(histFourS1GPU[kTimes], histFourS4GPU[kTimes], d_dirHist, workiir2, xsize, ysize, K, K2, pStream, 2 * PANGLE + 1);
			cudaDeviceSynchronize();
			cudaError_t cudastatus;
			cudastatus = cudaGetLastError();
			CHECK(cudastatus);


			int stx = xsize / 2;
			int sty = ysize / 2;
			int nMaxTheta;
			int maxITheta[2];


			for (pos = 0; pos < NMAXTHETA * GRIDSIZE * GRIDSIZE * 2; ++pos) {
				featuresTmpl->relativePosScL[pos] = (int)(scale * featuresTmpl->relativePosL[pos] + ROUNDFRAC);
				// printf("relativePosScL[%d] = %d\n", pos, featuresP->relativePosScL[pos]);
			}

			pos = stx + sty * xsize;

			/* Local direction using first mode */

			nMaxTheta = maxDirection(&(histFourS1GPU[kTimes][pos]), workMT, maxITheta);

			/* Rotate and construct feature vectors  */
			//printf("theta = %f, maxTheta = %f  %d  %f\n", 2.0 * PI * transType / NTEMPLATEROT, 2.0 * PI * maxITheta[0] / NMAXTHETA, nMaxTheta, 2.0 * PI * maxITheta[1] / NMAXTHETA);
			//printf("%f  %f \n", 2.0 * PI * transType / NTEMPLATEROT, 2.0 * PI * maxITheta[0] / NMAXTHETA);
					
			Feature *feature = &((featuresTmpl->featureL)[featuresTmpl->nFeature]);
			feature->ix = stx;
			feature->iy = sty;
			feature->iTheta = maxITheta[iFeature];
			feature->ordHist = iFeature;
			feature->scale = scale;
			feature->transType = 0;
			mkFeature(histFourS4GPU[kTimes], workMT, featuresTmpl, feature);
			++(featuresTmpl->nFeature);
					
			scale *= scaleRatio;
			kTimes++;

		}/* End of transformation loop */

		saveFeatures(fileName, featuresTmpl);

	}

	void MainProcessGPU::threadFunction(float scale, int KTime) {
		int pos, step, stX, stY, endX, endY, ix, iy, iFeature;
		int nMaxTheta;
		int maxITheta[2];


		for (pos = 0; pos < NMAXTHETA * GRIDSIZE * GRIDSIZE * 2; ++pos) {
			features[KTime]->relativePosScL[pos] = (int)(scale * features[KTime]->relativePosL[pos] + ROUNDFRAC);
			// printf("relativePosScL[%d] = %d\n", pos, featuresP->relativePosScL[pos]);
		}
		/* Process for each sample point */
		step = (int)(scale * STEPRATIO + ROUNDFRAC);


		stX = stY = (int)(1.5 * scale + ROUNDFRAC);
		endX = xsize - stX; endY = ysize - stY;


		for (iy = stY; iy <= endY; iy += step) {
			for (ix = stX; ix <= endX; ix += step) {
				// printf("(ix, iy) = (%d, %d)\n", ix, iy);
				pos = ix + xsize * iy;
				/* Local direction using first mode */

				nMaxTheta = maxDirection(&(histFourS1GPU[KTime][pos]), workMT, maxITheta);

				/* Rotate and construct feature vectors  */
				//printf("theta = %f, maxTheta = %f  %d  %f\n", 2.0 * PI * transType / NTEMPLATEROT, 2.0 * PI * maxITheta[0] / NMAXTHETA, nMaxTheta, 2.0 * PI * maxITheta[1] / NMAXTHETA);
				//printf("%f  %f \n", 2.0 * PI * transType / NTEMPLATEROT, 2.0 * PI * maxITheta[0] / NMAXTHETA);
				for (iFeature = 0; iFeature < nMaxTheta; ++iFeature) {
					Feature *feature = &((features[KTime]->featureL)[features[KTime]->nFeature]);
					feature->ix = ix;
					feature->iy = iy;
					feature->iTheta = maxITheta[iFeature];
					feature->ordHist = iFeature;
					feature->scale = scale;
					feature->transType = 0;
					mkFeature(histFourS4GPU[KTime], workMT, features[KTime], feature);
					++(features[KTime]->nFeature);
				}
			}
		} /* End of scale loop */
	}





#ifdef DEBUGMAIN
int main() {
	int nx = NX, ny = NY;
	int nxy = nx * ny;
	int *inImgI = (int*)malloc(nx*ny * sizeof(float));
	float *inImg = (float*)malloc(nx*ny * sizeof(float));
	initialApp(nx, ny, SCALEINIT, SCALEMAX);
	loadImageFile(fileName, inImgI, nx, ny);
	for (int ix = 0; ix < nxy; ++ix) inImg[ix] = inImgI[ix];

	calPoint(inImg, inImgI);
	return 0;
}

#endif // DEBUGMAIN


