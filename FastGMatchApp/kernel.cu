
#include "kernel.cuh"

#include "device_launch_parameters.h"
#include <math.h>

#include <stdio.h>
#include <stdlib.h>


#define ORIGINAL
//#define CHANGEPADDING



/*
 * calFeature.c
 *
 *  Created on: 2018/10/25
 *      Author: Morris Lee
 */
#define BDIMX 32
#define BDIMY BDIMX

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
__global__ void sincosIIRFGPU(float *d_intCos, float *d_intSin, float *d_sinL, float *d_cosL, float *d_paddingInImg, int nx, int ny, int K, int nOrd);
__global__ void gaussSmmothAfterIIR(float *d_blurImg, float *d_paddingInImg, float *d_intCos, float *d_blurCoef, int nOrd, int K, int nx, int ny);
__global__ void gaussDiffAfterIIR(float *d_diffImg, float *d_intSin, float *d_diffCoef, int nOrd, int K, int nx, int ny);
__global__ void transposeSmem(float *d_out, float *d_in, int nrows, int ncols);
__global__ void paddingZeroGpu(float *d_outImg, float *d_inImg, const int nx, const int ny, const int K);
__global__ void paddingNonzeroGpu(float *d_outImg, float *d_inImg, const int nx, const int ny, const int K);
void paddingGpuAsync(float *d_outImg, float *d_inImg, int nx, int ny, int K, int flag, cudaStream_t pStream);
void paddingGpu(float *d_outImg, float *d_inImg, int nx, int ny, int K, int flag);
__global__ void calDirHistPoint2(const FPTYPE *diffXImg, const FPTYPE *diffYImg, FPTYPE *dirHist, const int nx, const int ny);

/* Coefficients P = 2, sigma = 1.1 */
__constant__ float coefG2G[] = { 1.5846315202e-01, 1.7508079293e-01, 2.7323762984e-02 };
__constant__ float coefDG2G[] = { 0.0000000000e+00, -1.7506270969e-01, -5.4683692457e-02 };

/* Coefficients P = 4, sigma = 0.8 */
__constant__ float coefG4G[] = { 1.5914081930e-01,  2.3116780826e-01,  8.8476942331e-02,  1.7890179087e-02,  1.8836364529e-03 };
__constant__ float coefDG4G[] = { 0.0000000000e+00, -2.3116693143e-01, -1.7695563833e-01, -5.3667906760e-02, -7.5380531476e-03 };


/*initial the sin and cos number by K*/
__global__ void initialByK(float *cosL, float *sinL, int K, int nOrd) {
	int order;
	for (order = 0; order <= nOrd; ++order) {
		cosL[order] = cos(PI * order / K);
		sinL[order] = sin(PI * order / K);
	}
}

/*initial the sin and cos coef by ord work for ord 4*/
__global__ void initialCoef4(float *blurCoef, float *diffCoef, const int nOrd) {
	float sign = 1.0;
	for (int pos = 0; pos <= nOrd; ++pos) {
		blurCoef[pos] = sign * coefG4G[pos];
		diffCoef[pos] = sign * coefDG4G[pos];
		blurCoef[nOrd + 1] += blurCoef[pos];
		sign *= -1;
	}
}

/*initial the sin and cos coef by ord work for ord 2*/
__global__ void initialCoef2(float *blurCoef, float *diffCoef, const int nOrd) {
	float sign = 1.0;
	for (int pos = 0; pos <= nOrd; ++pos) {
		blurCoef[pos] = sign * coefG2G[pos];
		diffCoef[pos] = sign * coefDG2G[pos];
		blurCoef[nOrd + 1] += blurCoef[pos];
		sign *= -1;
	}
}


/*initial the workspace for ord 4*/
void initialGPU4(WorkIIRGPU *workIIR4, int nx, int ny, int nOrd,int maxK) {
	workIIR4->nOrd = nOrd;
	int maxsize = nx > ny ? nx : ny;
	int minsize = nx > ny ? ny : nx;
	CHECK(cudaMalloc((void**)&(workIIR4->d_cosL), sizeof(float)*workIIR4->nOrd + 1));
	CHECK(cudaMalloc((void**)&(workIIR4->d_sinL), sizeof(float)*workIIR4->nOrd + 1));
	CHECK(cudaMalloc((void**)&(workIIR4->d_blurCoef), sizeof(float)*(workIIR4->nOrd + 2)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_diffCoef), sizeof(float)*(workIIR4->nOrd + 1)));
	initialCoef4 << <1, 1 >> > (workIIR4->d_blurCoef, workIIR4->d_diffCoef, workIIR4->nOrd);
	CHECK(cudaMalloc((void**)&(workIIR4->d_intSin), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_intCos), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_intSinDiff), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_intCosDiff), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_tranIIRSin), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_tranIIRCos), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR4->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_inImg), sizeof(float)*ny*nx));
	CHECK(cudaMalloc((void**)&(workIIR4->d_paddingImg), sizeof(float)*maxsize*(minsize + 2 * maxK)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_paddingImgdiff), sizeof(float)*maxsize*(minsize + 2 * maxK)));
	CHECK(cudaMalloc((void**)&(workIIR4->d_tranblurImg), sizeof(float)*ny*nx));
	CHECK(cudaMalloc((void**)&(workIIR4->d_trandiffImg), sizeof(float)*ny*nx));
	CHECK(cudaMalloc((void**)&(workIIR4->d_blurImg), sizeof(float)*ny*nx));
	CHECK(cudaMalloc((void**)&(workIIR4->d_diffImg), sizeof(float)*ny*nx));

	cudaDeviceSynchronize();
}

/*delete the workspace for ord 4*/
void deleteGPU4(WorkIIRGPU *workIIR4) {

	cudaFree(workIIR4->d_cosL);
	cudaFree(workIIR4->d_sinL);
	cudaFree(workIIR4->d_blurCoef);
	cudaFree(workIIR4->d_diffCoef);
	cudaFree(workIIR4->d_intSin);
	cudaFree(workIIR4->d_intCos);
	cudaFree(workIIR4->d_intSinDiff);
	cudaFree(workIIR4->d_intCosDiff);
	cudaFree(workIIR4->d_tranIIRSin);
	cudaFree(workIIR4->d_tranIIRCos);
	cudaFree(workIIR4->d_inImg);
	cudaFree(workIIR4->d_paddingImg);
	cudaFree(workIIR4->d_paddingImgdiff);
	cudaFree(workIIR4->d_tranblurImg);
	cudaFree(workIIR4->d_trandiffImg);
	cudaFree(workIIR4->d_blurImg);
	cudaFree(workIIR4->d_diffImg);

}

/*initial the workspace for ord 2*/
void initialGPU2(WorkIIRGPU *workIIR2, int nx, int ny, int nOrd,int maxK) {
	workIIR2->nOrd = nOrd;
	int maxsize = nx > ny ? nx : ny;
	int minsize = nx > ny ? ny : nx;
	CHECK(cudaMalloc((void**)&(workIIR2->d_cosL), sizeof(float)*workIIR2->nOrd + 1));
	CHECK(cudaMalloc((void**)&(workIIR2->d_sinL), sizeof(float)*workIIR2->nOrd + 1));
	CHECK(cudaMalloc((void**)&(workIIR2->d_blurCoef), sizeof(float)*(workIIR2->nOrd + 2)));
	CHECK(cudaMalloc((void**)&(workIIR2->d_diffCoef), sizeof(float)*(workIIR2->nOrd + 1)));
	initialCoef2 << <1, 1 >> > (workIIR2->d_blurCoef, workIIR2->d_diffCoef, workIIR2->nOrd);
	CHECK(cudaMalloc((void**)&(workIIR2->d_intSin), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR2->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR2->d_intCos), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR2->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR2->d_tranIIRCos), sizeof(float)*maxsize*(minsize + 2 * maxK)*(workIIR2->nOrd + 1)));
	CHECK(cudaMalloc((void**)&(workIIR2->d_paddingImg), sizeof(float)*maxsize*(minsize + 2 * maxK)));
	CHECK(cudaMalloc((void**)&(workIIR2->d_tranblurImg), sizeof(float)*ny*nx));
	CHECK(cudaMalloc((void**)&(workIIR2->d_blurImg), sizeof(float)*ny*nx));

}

/*delete the workspace for ord 2*/
void deleteGPU2(WorkIIRGPU *workIIR2) {

	cudaFree(workIIR2->d_cosL);
	cudaFree(workIIR2->d_sinL);
	cudaFree(workIIR2->d_blurCoef);
	cudaFree(workIIR2->d_diffCoef);
	cudaFree(workIIR2->d_intSin);
	cudaFree(workIIR2->d_intCos);
	cudaFree(workIIR2->d_tranIIRCos);
	cudaFree(workIIR2->d_inImg);
	cudaFree(workIIR2->d_paddingImg);
	cudaFree(workIIR2->d_tranblurImg);
	cudaFree(workIIR2->d_blurImg);

}

/*asynchronize gauss smooth using gpu*/
void gaussSmoothGPUAsync(float *blurImg, float*inImg, WorkIIRGPU *workiir, int nx, int ny, int K, cudaStream_t pStream) {
	if (K < 2)K = 2;

	int nxy = ny * nx;
	//CHECK(cudaMemcpyAsync(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice, pStream));
	initialByK << <1, 1, 0, pStream >> > (workiir->d_cosL, workiir->d_sinL, K, workiir->nOrd);
	paddingGpuAsync(workiir->d_paddingImg, inImg, nx, ny, K, 1, pStream);

	int maxsize = nx > ny ? nx : ny;
	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 block2(maxsize);
	dim3 grid2((workiir->nOrd + 1));
	dim3 block3(BDIMX, BDIMY);
	dim3 grid3((ny*(workiir->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K + block.y - 1) / block.y);

	dim3 block4(BDIMX, BDIMY);
	dim3 grid4((ny + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
	dim3 block5(BDIMX, BDIMY);
	dim3 grid5((nx*(workiir->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K + block.y - 1) / block.y);

	sincosIIRFGPU << <grid2, block2, 0, pStream >> > (workiir->d_intCos, workiir->d_intSin, workiir->d_sinL, workiir->d_cosL, workiir->d_paddingImg, nx, ny, K, workiir->nOrd);
	transposeSmem << <grid3, block3, 0, pStream >> > (workiir->d_tranIIRCos, workiir->d_intCos, nx + 2 * K, ny*(workiir->nOrd + 1));
	gaussSmmothAfterIIR << <grid, block, 0, pStream >> > (workiir->d_blurImg, workiir->d_paddingImg, workiir->d_tranIIRCos, workiir->d_blurCoef, workiir->nOrd, K, nx, ny);
	transposeSmem << <grid, block, 0, pStream >> > (workiir->d_tranblurImg, workiir->d_blurImg, ny, nx);
	paddingGpuAsync(workiir->d_paddingImg, workiir->d_tranblurImg, ny, nx, K, 1, pStream);
	sincosIIRFGPU << <grid2, block2, 0, pStream >> > (workiir->d_intCos, workiir->d_intSin, workiir->d_sinL, workiir->d_cosL, workiir->d_paddingImg, ny, nx, K, workiir->nOrd);
	transposeSmem << <grid5, block5, 0, pStream >> > (workiir->d_tranIIRCos, workiir->d_intCos, ny + 2 * K, nx*(workiir->nOrd + 1));
	gaussSmmothAfterIIR << <grid4, block4, 0, pStream >> > (workiir->d_tranblurImg, workiir->d_paddingImg, workiir->d_tranIIRCos, workiir->d_blurCoef, workiir->nOrd, K, ny, nx);
	transposeSmem << <grid4, block4, 0, pStream >> > (workiir->d_blurImg, workiir->d_tranblurImg, nx, ny);

	cudaMemcpyAsync(blurImg, workiir->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost, pStream);
}

void gaussSmoothGPUAsyncWidth(float *blurImg, float *blurImg2, float*inImg, WorkIIRGPU **workiir, int nx, int ny, int K, int K2, cudaStream_t *pStream, int times) {
	if (K < 2)K = 2;
	int i;
	int nxy = ny * nx;
	//CHECK(cudaMemcpyAsync(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice, pStream));
	initialByK << <1, 1 >> > (workiir[0]->d_cosL, workiir[0]->d_sinL, K, workiir[0]->nOrd);
	initialByK << <1, 1 >> > (workiir[times]->d_cosL, workiir[times]->d_sinL, K2, workiir[times]->nOrd);

	for (i = 0; i < times; i++) {
		paddingGpuAsync(workiir[i]->d_paddingImg, (inImg + i * nxy), nx, ny, K, 1, pStream[i]);
		paddingGpuAsync(workiir[i + times]->d_paddingImg, (inImg + i * nxy), nx, ny, K2, 1, pStream[i + times]);
	}

	int maxsize = nx > ny ? nx : ny;
	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 block2(maxsize);
	dim3 grid2((workiir[0]->nOrd + 1));
	dim3 block3(BDIMX, BDIMY);
	dim3 grid3((ny*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K + block.y - 1) / block.y);
	dim3 block6(BDIMX, BDIMY);
	dim3 grid6((ny*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K2 + block.y - 1) / block.y);

	dim3 block4(BDIMX, BDIMY);
	dim3 grid4((ny + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
	dim3 block5(BDIMX, BDIMY);
	dim3 grid5((nx*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K + block.y - 1) / block.y);
	dim3 block7(BDIMX, BDIMY);
	dim3 grid7((nx*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K2 + block.y - 1) / block.y);

	for (i = 0; i < times; i++) {
		sincosIIRFGPU << <grid2, block2, 0, pStream[i] >> > (workiir[i]->d_intCos, workiir[i]->d_intSin, workiir[0]->d_sinL, workiir[0]->d_cosL, workiir[i]->d_paddingImg, nx, ny, K, workiir[i]->nOrd);
		sincosIIRFGPU << <grid2, block2, 0, pStream[i + times] >> > (workiir[i + times]->d_intCos, workiir[i + times]->d_intSin, workiir[times]->d_sinL, workiir[times]->d_cosL, workiir[i + times]->d_paddingImg, nx, ny, K2, workiir[i + times]->nOrd);

	}
	for (i = 0; i < times; i++) {
		transposeSmem << <grid3, block3, 0, pStream[i] >> > (workiir[i]->d_tranIIRCos, workiir[i]->d_intCos, nx + 2 * K, ny*(workiir[i]->nOrd + 1));
		transposeSmem << <grid6, block6, 0, pStream[i + times] >> > (workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_intCos, nx + 2 * K2, ny*(workiir[i + times]->nOrd + 1));

	}
	for (i = 0; i < times; i++) {
		gaussSmmothAfterIIR << <grid, block, 0, pStream[i] >> > (workiir[i]->d_blurImg, workiir[i]->d_paddingImg, workiir[i]->d_tranIIRCos, workiir[i]->d_blurCoef, workiir[i]->nOrd, K, nx, ny);
		gaussSmmothAfterIIR << <grid, block, 0, pStream[i + times] >> > (workiir[i + times]->d_blurImg, workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_blurCoef, workiir[i + times]->nOrd, K2, nx, ny);

	}
	for (i = 0; i < times; i++) {
		transposeSmem << <grid, block, 0, pStream[i] >> > (workiir[i]->d_tranblurImg, workiir[i]->d_blurImg, ny, nx);
		transposeSmem << <grid, block, 0, pStream[i + times] >> > (workiir[i + times]->d_tranblurImg, workiir[i + times]->d_blurImg, ny, nx);

	}
	for (i = 0; i < times; i++) {
		paddingGpuAsync(workiir[i]->d_paddingImg, workiir[i]->d_tranblurImg, ny, nx, K, 1, pStream[i]);
		paddingGpuAsync(workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranblurImg, ny, nx, K2, 1, pStream[i + times]);

	}
	for (i = 0; i < times; i++) {
		sincosIIRFGPU << <grid2, block2, 0, pStream[i] >> > (workiir[i]->d_intCos, workiir[i]->d_intSin, workiir[0]->d_sinL, workiir[0]->d_cosL, workiir[i]->d_paddingImg, ny, nx, K, workiir[i]->nOrd);
		sincosIIRFGPU << <grid2, block2, 0, pStream[i + times] >> > (workiir[i + times]->d_intCos, workiir[i + times]->d_intSin, workiir[times]->d_sinL, workiir[times]->d_cosL, workiir[i + times]->d_paddingImg, ny, nx, K2, workiir[i + times]->nOrd);

	}
	for (i = 0; i < times; i++) {
		transposeSmem << <grid5, block5, 0, pStream[i] >> > (workiir[i]->d_tranIIRCos, workiir[i]->d_intCos, ny + 2 * K, nx*(workiir[i]->nOrd + 1));
		transposeSmem << <grid7, block7, 0, pStream[i + times] >> > (workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_intCos, ny + 2 * K2, nx*(workiir[i + times]->nOrd + 1));

	}
	for (i = 0; i < times; i++) {
		gaussSmmothAfterIIR << <grid4, block4, 0, pStream[i] >> > (workiir[i]->d_tranblurImg, workiir[i]->d_paddingImg, workiir[i]->d_tranIIRCos, workiir[i]->d_blurCoef, workiir[i]->nOrd, K, ny, nx);
		gaussSmmothAfterIIR << <grid4, block4, 0, pStream[i + times] >> > (workiir[i + times]->d_tranblurImg, workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_blurCoef, workiir[i + times]->nOrd, K2, ny, nx);

	}
	for (i = 0; i < times; i++) {
		transposeSmem << <grid4, block4, 0, pStream[i] >> > (workiir[i]->d_blurImg, workiir[i]->d_tranblurImg, nx, ny);
		transposeSmem << <grid4, block4, 0, pStream[i + times] >> > (workiir[i + times]->d_blurImg, workiir[i + times]->d_tranblurImg, nx, ny);

	}
	for (i = 0; i < times; i++) {
		cudaMemcpyAsync((blurImg + i * nxy), workiir[i]->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost, pStream[i]);
		cudaMemcpyAsync((blurImg2 + i * nxy), workiir[i + times]->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost, pStream[i + times]);

	}


}

void gaussSmoothGPUAsyncHeight(float *blurImg, float *blurImg2, float*inImg, WorkIIRGPU **workiir, int nx, int ny, int K, int K2, cudaStream_t *pStream, int times) {
	if (K < 2)K = 2;
	int i;
	int nxy = ny * nx;
	//CHECK(cudaMemcpyAsync(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice, pStream));
	initialByK << <1, 1 >> > (workiir[0]->d_cosL, workiir[0]->d_sinL, K, workiir[0]->nOrd);
	initialByK << <1, 1 >> > (workiir[times]->d_cosL, workiir[times]->d_sinL, K2, workiir[times]->nOrd);

	int maxsize = nx > ny ? nx : ny;
	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 block2(maxsize);
	dim3 grid2((workiir[0]->nOrd + 1));
	dim3 block3(BDIMX, BDIMY);
	dim3 grid3((ny*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K + block.y - 1) / block.y);
	dim3 block6(BDIMX, BDIMY);
	dim3 grid6((ny*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K2 + block.y - 1) / block.y);

	dim3 block4(BDIMX, BDIMY);
	dim3 grid4((ny + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
	dim3 block5(BDIMX, BDIMY);
	dim3 grid5((nx*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K + block.y - 1) / block.y);
	dim3 block7(BDIMX, BDIMY);
	dim3 grid7((nx*(workiir[0]->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K2 + block.y - 1) / block.y);

	for (i = 0; i < times; i++) {
		paddingGpuAsync(workiir[i]->d_paddingImg, (inImg + i * nxy), nx, ny, K, 1, pStream[i]);
		paddingGpuAsync(workiir[i + times]->d_paddingImg, (inImg + i * nxy), nx, ny, K2, 1, pStream[i + times]);

		sincosIIRFGPU << <grid2, block2, 0, pStream[i] >> > (workiir[i]->d_intCos, workiir[i]->d_intSin, workiir[0]->d_sinL, workiir[0]->d_cosL, workiir[i]->d_paddingImg, nx, ny, K, workiir[i]->nOrd);
		sincosIIRFGPU << <grid2, block2, 0, pStream[i + times] >> > (workiir[i + times]->d_intCos, workiir[i + times]->d_intSin, workiir[times]->d_sinL, workiir[times]->d_cosL, workiir[i + times]->d_paddingImg, nx, ny, K2, workiir[i + times]->nOrd);


		transposeSmem << <grid3, block3, 0, pStream[i] >> > (workiir[i]->d_tranIIRCos, workiir[i]->d_intCos, nx + 2 * K, ny*(workiir[i]->nOrd + 1));
		transposeSmem << <grid6, block6, 0, pStream[i + times] >> > (workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_intCos, nx + 2 * K2, ny*(workiir[i + times]->nOrd + 1));


		gaussSmmothAfterIIR << <grid, block, 0, pStream[i] >> > (workiir[i]->d_blurImg, workiir[i]->d_paddingImg, workiir[i]->d_tranIIRCos, workiir[i]->d_blurCoef, workiir[i]->nOrd, K, nx, ny);
		gaussSmmothAfterIIR << <grid, block, 0, pStream[i + times] >> > (workiir[i + times]->d_blurImg, workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_blurCoef, workiir[i + times]->nOrd, K2, nx, ny);


		transposeSmem << <grid, block, 0, pStream[i] >> > (workiir[i]->d_tranblurImg, workiir[i]->d_blurImg, ny, nx);
		transposeSmem << <grid, block, 0, pStream[i + times] >> > (workiir[i + times]->d_tranblurImg, workiir[i + times]->d_blurImg, ny, nx);


		paddingGpuAsync(workiir[i]->d_paddingImg, workiir[i]->d_tranblurImg, ny, nx, K, 1, pStream[i]);
		paddingGpuAsync(workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranblurImg, ny, nx, K2, 1, pStream[i + times]);


		sincosIIRFGPU << <grid2, block2, 0, pStream[i] >> > (workiir[i]->d_intCos, workiir[i]->d_intSin, workiir[0]->d_sinL, workiir[0]->d_cosL, workiir[i]->d_paddingImg, ny, nx, K, workiir[i]->nOrd);
		sincosIIRFGPU << <grid2, block2, 0, pStream[i + times] >> > (workiir[i + times]->d_intCos, workiir[i + times]->d_intSin, workiir[times]->d_sinL, workiir[times]->d_cosL, workiir[i + times]->d_paddingImg, ny, nx, K2, workiir[i + times]->nOrd);


		transposeSmem << <grid5, block5, 0, pStream[i] >> > (workiir[i]->d_tranIIRCos, workiir[i]->d_intCos, ny + 2 * K, nx*(workiir[i]->nOrd + 1));
		transposeSmem << <grid7, block7, 0, pStream[i + times] >> > (workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_intCos, ny + 2 * K2, nx*(workiir[i + times]->nOrd + 1));


		gaussSmmothAfterIIR << <grid4, block4, 0, pStream[i] >> > (workiir[i]->d_tranblurImg, workiir[i]->d_paddingImg, workiir[i]->d_tranIIRCos, workiir[i]->d_blurCoef, workiir[i]->nOrd, K, ny, nx);
		gaussSmmothAfterIIR << <grid4, block4, 0, pStream[i + times] >> > (workiir[i + times]->d_tranblurImg, workiir[i + times]->d_paddingImg, workiir[i + times]->d_tranIIRCos, workiir[i + times]->d_blurCoef, workiir[i + times]->nOrd, K2, ny, nx);


		transposeSmem << <grid4, block4, 0, pStream[i] >> > (workiir[i]->d_blurImg, workiir[i]->d_tranblurImg, nx, ny);
		transposeSmem << <grid4, block4, 0, pStream[i + times] >> > (workiir[i + times]->d_blurImg, workiir[i + times]->d_tranblurImg, nx, ny);

		cudaMemcpyAsync((blurImg + i * nxy), workiir[i]->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost, pStream[i]);
		cudaMemcpyAsync((blurImg2 + i * nxy), workiir[i + times]->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost, pStream[i + times]);

	}


}


/*asynchronize gauss diff using gpu*/
void gaussDiffGPUAsync(float *diffXImg, float *diffYImg, float *inImg, WorkIIRGPU *workiir, int nx, int ny, int K, cudaStream_t *pStream) {
	if (K < 2)K = 2;
	initialByK << <1, 1 >> > (workiir->d_cosL, workiir->d_sinL, K, workiir->nOrd);
	//int nxy = ny * nx;
	//CHECK(cudaMemcpy(workiir->d_inImg, inImg, nxy * sizeof(float), cudaMemcpyHostToDevice));

	paddingGpu(workiir->d_paddingImg, workiir->d_inImg, nx, ny, K, 1);

	int maxsize = nx > ny ? nx : ny;
	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	dim3 block2(maxsize);
	dim3 grid2((workiir->nOrd + 1));
	dim3 block3(BDIMX, BDIMY);
	dim3 grid3((ny*(workiir->nOrd + 1) + block.x - 1) / block.x, (nx + 2 * K + block.y - 1) / block.y);

	dim3 block4(BDIMX, BDIMY);
	dim3 grid4((ny + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
	dim3 block5(BDIMX, BDIMY);
	dim3 grid5((nx*(workiir->nOrd + 1) + block.x - 1) / block.x, (ny + 2 * K + block.y - 1) / block.y);


	sincosIIRFGPU << <grid2, block2 >> > (workiir->d_intCos, workiir->d_intSin, workiir->d_sinL, workiir->d_cosL, workiir->d_paddingImg, nx, ny, K, workiir->nOrd);

	transposeSmem << <grid3, block3, 0, pStream[0] >> > (workiir->d_tranIIRCos, workiir->d_intCos, nx + 2 * K, ny*(workiir->nOrd + 1));
	transposeSmem << <grid3, block3, 0, pStream[1] >> > (workiir->d_tranIIRSin, workiir->d_intSin, nx + 2 * K, ny*(workiir->nOrd + 1));


	gaussSmmothAfterIIR << <grid, block, 0, pStream[0] >> > (workiir->d_blurImg, workiir->d_paddingImg, workiir->d_tranIIRCos, workiir->d_blurCoef, workiir->nOrd, K, nx, ny);
	gaussDiffAfterIIR << <grid, block, 0, pStream[1] >> > (workiir->d_diffImg, workiir->d_tranIIRSin, workiir->d_diffCoef, workiir->nOrd, K, nx, ny);

	transposeSmem << <grid, block, 0, pStream[0] >> > (workiir->d_tranblurImg, workiir->d_blurImg, ny, nx);
	transposeSmem << <grid, block, 0, pStream[1] >> > (workiir->d_trandiffImg, workiir->d_diffImg, ny, nx);

	paddingGpuAsync(workiir->d_paddingImg, workiir->d_tranblurImg, ny, nx, K, 1, pStream[0]);
	paddingGpuAsync(workiir->d_paddingImgdiff, workiir->d_trandiffImg, ny, nx, K, 1, pStream[1]);

	sincosIIRFGPU << <grid2, block2, 0, pStream[0] >> > (workiir->d_intCos, workiir->d_intSin, workiir->d_sinL, workiir->d_cosL, workiir->d_paddingImg, ny, nx, K, workiir->nOrd);
	sincosIIRFGPU << <grid2, block2, 0, pStream[1] >> > (workiir->d_intCosDiff, workiir->d_intSinDiff, workiir->d_sinL, workiir->d_cosL, workiir->d_paddingImgdiff, ny, nx, K, workiir->nOrd);

	transposeSmem << <grid5, block5, 0, pStream[0] >> > (workiir->d_tranIIRSin, workiir->d_intSin, ny + 2 * K, nx*(workiir->nOrd + 1));
	transposeSmem << <grid5, block5, 0, pStream[1] >> > (workiir->d_tranIIRCos, workiir->d_intCosDiff, ny + 2 * K, nx*(workiir->nOrd + 1));

	gaussDiffAfterIIR << <grid4, block4, 0, pStream[0] >> > (workiir->d_tranblurImg, workiir->d_tranIIRSin, workiir->d_diffCoef, workiir->nOrd, K, ny, nx);
	gaussSmmothAfterIIR << <grid4, block4, 0, pStream[1] >> > (workiir->d_trandiffImg, workiir->d_paddingImgdiff, workiir->d_tranIIRCos, workiir->d_blurCoef, workiir->nOrd, K, ny, nx);


	transposeSmem << <grid4, block4, 0, pStream[0] >> > (workiir->d_blurImg, workiir->d_tranblurImg, nx, ny);
	transposeSmem << <grid4, block4, 0, pStream[1] >> > (workiir->d_diffImg, workiir->d_trandiffImg, nx, ny);

	//cudaMemcpy(diffYImg, workiir->d_blurImg, nxy * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(diffXImg, workiir->d_diffImg, nxy * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
}

/* Make feature vector */
int mkFeature(FPTYPE *histFourS4, WorkMT *workMT, Features *features, Feature *feature) {
	int ix, iy, ifx, ify, pos, order, posXY, posXY2, posHist;
	double sqSum, sqr;
	int *rePosSc;

	// printf("iTheta = %d\n", feature->iTheta);
	rePosSc = &(features->relativePosScL[feature->iTheta * GRIDSIZE * GRIDSIZE * 2]);

	posXY = posXY2 = 0;
	for (ify = 0; ify < GRIDSIZE; ++ify) {
		for (ifx = 0; ifx < GRIDSIZE; ++ifx) {
			//printf("(%d, %d)  ", rePosSc[posXY2], rePosSc[posXY2 + 1]);
			ix = feature->ix + rePosSc[posXY2++];
			iy = feature->iy + rePosSc[posXY2++];
			// printf("(%d, %d)\n", ix, iy);
			posHist = ix + features->nx * iy;
			features->tmpVector[0] = histFourS4[posHist];
			for (order = 1; order < features->nAngleCoef; ++order) {
				posHist += features->nxy;
				features->tmpVector[order] = histFourS4[posHist];
			}
			rotateDirHist(features->tmpVector, &(feature->vector[features->nAngleCoef * (posXY++)]), feature->iTheta * features->nAngleCoef, workMT);
			//for (ix = 0 ; ix < features->nAngleCoef ; ++ix) printf("%f, ", feature->vector[features->nAngleCoef * (posXY - 1) + ix]);
			//printf("\n");
		}
	}
	/* Normalization */
	sqSum = 0.0;
	for (pos = 0; pos < features->nAngleCoef * GRIDSIZE * GRIDSIZE; ++pos) 	sqSum += feature->vector[pos] * feature->vector[pos];
	sqr = sqrt(sqSum);
	for (pos = 0; pos < features->nAngleCoef * GRIDSIZE * GRIDSIZE; ++pos) feature->vector[pos] /= sqr;
	return 0;
}


/* Calculate Fourier expression of directional histogram */
int calDirHistPointGPU(FPTYPE *d_diffXImg, FPTYPE *d_diffYImg, FPTYPE *d_dirHist, FPTYPE *dirHist, int nx, int ny) {




	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	calDirHistPoint2 << <grid, block >> > (d_diffXImg, d_diffYImg, d_dirHist, nx, ny);
	cudaError_t cudastatus;
	cudastatus = cudaGetLastError();
	CHECK(cudastatus);
	int  nxy = nx * ny;
	int nAngleCoef = 2 * PANGLE + 1;
	size_t nAngleBytes = nxy * nAngleCoef * sizeof(FPTYPE);
	CHECK(cudaMemcpy(dirHist, d_dirHist, nAngleBytes, cudaMemcpyDeviceToHost));
	return 0;
}

/* Calculate Fourier expression of directional histogram asynchronizedly*/
int calDirHistPointGPUAsync(FPTYPE *d_diffXImg, FPTYPE *d_diffYImg, FPTYPE *d_dirHist, int nx, int ny) {




	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	calDirHistPoint2 << <grid, block >> > (d_diffXImg, d_diffYImg, d_dirHist, nx, ny);
	//cudaError_t cudastatus;
	//cudastatus = cudaGetLastError();
	//CHECK(cudastatus);
	return 0;
}

/*gpu code to calculate Fourier expression of directional histogram*/
__global__ void calDirHistPoint2(const FPTYPE *diffXImg, const FPTYPE *diffYImg, FPTYPE *dirHist, const int nx, const int ny) {
	unsigned int nxy = nx * ny, nxy2 = 2 * nxy;
	unsigned int stCos = nxy, stSin = nxy2;
	//unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x + blockIdx.y*blockDim.y+blockDim.x*gridDim.x*threadIdx.y;
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int pos = iy * gridDim.x*blockDim.x + ix;
	if (pos >= nxy)return;
	FPTYPE ansCos, ansSin, diffCos, diffSin;
	FPTYPE diffX = diffXImg[pos];
	FPTYPE diffY = diffYImg[pos];
	FPTYPE diffPower = diffX * diffX + diffY * diffY;
	FPTYPE diffSqrt;
	dirHist[pos] = diffSqrt = sqrt(diffPower + EDGEPOWER);
	dirHist[pos + stCos] = ansCos = diffX;
	dirHist[pos + stSin] = ansSin = -diffY;
	diffCos = diffX / diffSqrt;
	diffSin = diffY / diffSqrt;
	for (int order = 2; order <= PANGLE; ++order) {
		stCos = stCos + nxy2, stSin = stSin + nxy2;
		dirHist[pos + stCos] = ansCos * diffCos + ansSin * diffSin;
		dirHist[pos + stSin] = ansSin = -ansCos * diffSin + ansSin * diffCos;
		ansCos = dirHist[pos + stCos];
	}

}

/* Search direction with the max value in histogram */
int maxDirection(FPTYPE *histFourS1, WorkMT *workMT, int *maxITheta) {
	int nMaxTheta = 0, iTheta, iThetaP, order, posCoef;
	int nData = workMT->nx * workMT->ny;
	FPTYPE vals[NMAXTHETA + 2], val, maxVal1, maxVal2;
	int pos, maxPos1, maxPos2;
	pos = 0;
	for (iTheta = 0; iTheta < NMAXTHETA; ++iTheta) {
		val = 0.0;
		posCoef = 0;
		for (order = 0; order < 2 * PANGLE + 1; ++order) {
			val += workMT->invFourTbl[pos++] * histFourS1[posCoef];
			//printf("%d  %f\n", order, blurLargeDirHist[posCoef]);
			posCoef += nData;
		}
		vals[iTheta + 1] = val;
		// printf("iTheta = %d val = %f \n", iTheta, vals[iTheta + 1]);
	}
	vals[0] = vals[NMAXTHETA];
	vals[NMAXTHETA + 1] = vals[1];
	maxPos1 = maxPos2 = 0;
	for (iThetaP = 1; iThetaP <= NMAXTHETA; ++iThetaP) {
		if (vals[iThetaP] > vals[iThetaP - 1] && vals[iThetaP] > vals[iThetaP + 1]) { /* Local maximum */
			switch (nMaxTheta) {
			case 0:
				maxVal1 = vals[iThetaP];
				maxPos1 = iThetaP;
				nMaxTheta = 1;
				break;
			case 1:
				if (vals[iThetaP] > maxVal1) {
					maxVal2 = maxVal1;
					maxPos2 = maxPos1;
					maxVal1 = vals[iThetaP];
					maxPos1 = iThetaP;
				}
				else {
					maxVal2 = vals[iThetaP];
					maxPos2 = iThetaP;
				}
				nMaxTheta = 2;
				break;
			case 2:
				if (vals[iThetaP] > maxVal1) {
					maxVal2 = maxVal1;
					maxPos2 = maxPos1;
					maxVal1 = vals[iThetaP];
					maxPos1 = iThetaP;
				}
				else if (vals[iThetaP] > maxVal2) {
					maxVal2 = vals[iThetaP];
					maxPos2 = iThetaP;
				}
				break;
			}
		}
	}

	if (nMaxTheta == 2 && maxVal2 < maxVal1 * 0.7) nMaxTheta = 1;
	maxITheta[0] = maxPos1 - 1;
	maxITheta[1] = maxPos2 - 1;
	// printf("nMaxTheta = %d, maxITheta1 = %d maxTheta2 = %d  maxVal1 = %.1f maxVal2 = %.1f \n", nMaxTheta, maxITheta[0], maxITheta[1], maxVal1,  maxVal2 );
	return nMaxTheta;
}

/* Rotation of directional histogram */
int rotateDirHist(FPTYPE *vectorIn, FPTYPE *vectorOut, int stRotTbl, WorkMT *workMT) {
	int order, posRotTbl, posVect;
	FPTYPE vCos, vSin, rCos, rSin;

	posRotTbl = stRotTbl + 1;
	posVect = 0;
	vectorOut[posVect] = vectorIn[posVect]; ++posVect;
	for (order = 1; order <= PANGLE; ++order) {
		vCos = vectorIn[posVect];
		vSin = vectorIn[posVect + 1];
		rCos = workMT->rotTbl[posRotTbl++];
		rSin = workMT->rotTbl[posRotTbl++];
		vectorOut[posVect++] = vCos * rCos - vSin * rSin;
		vectorOut[posVect++] = vCos * rSin + vSin * rCos;
	}
	return 0;
}

/* Large scale smoothing is approximated by small scale smoothing */
int approxLargeScale(FPTYPE *histFourS4, FPTYPE *histFourAS1, int ix, int iy, WorkMT *workMT) {
	int pInd, pos, tPos, tPos2, order, orderP;
	int nxy = workMT->nx * workMT->ny;
	FPTYPE a;

	/* Initialize the output */
	for (order = 0; order < 2 * PANGLE + 1; ++order) 	histFourAS1[order] = 0.0;

	tPos = tPos2 = 0;
	for (pInd = 0; pInd < NAPPROPOINT; ++pInd) {
		//printf("(%d, %d) (%d, %d) a = %f \n", ix, iy, ix + workMT->largeScaleTbl[tPos2], iy + workMT->largeScaleTbl[tPos2+1], workMT->largeScaleATbl[tPos]);
		pos = (ix + workMT->largeScaleTbl[tPos2]);
		pos += workMT->nx * (iy + workMT->largeScaleTbl[tPos2++]);
		a = workMT->largeScaleATbl[tPos++];
		orderP = 0;
		for (order = 0; order < 2 * PANGLE + 1; ++order) {
			//printf("orderP = %d, pos = %d \n", orderP, pos);
			histFourAS1[orderP] += a * histFourS4[pos];
			pos += nxy; orderP += nxy;
		}
	}
	//for (order = 0 ; order < 2 * PANGLE + 1 ; ++order) printf("order %d %lf\n", order,	outHist[order]);
	return 0;
}

//memoric
__global__ void sincosIIRFGPU(float *d_intCos, float *d_intSin, float *d_sinL, float *d_cosL, float *d_paddingInImg, int nx, int ny, int K, int nOrd) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int id = idx + idy * gridDim.x * blockDim.x;










#ifdef ORIGINAL
	if (id >= ny * (nOrd + 1)) return;
	unsigned int original = id / (nOrd + 1);
	unsigned int nInt = (nx + 2 * K);
	unsigned int shiftImg = original * nInt;
	unsigned int nIntCos = ny * (nOrd + 1);
#else
#ifdef CHANGEPADDING
	if (id >= nx * (nOrd + 1)) return;
	unsigned int shiftImg = id % nx;
	unsigned int nInt = (ny + 2 * K);
	unsigned int nIntCos = nx * (nOrd + 1);
#endif // CHANGEPADDING

#endif // ORIGINAL
	unsigned int loopOrd = id % (nOrd + 1);


	unsigned int posInt1;
	float valM;
	float valC;
	float valS;
	unsigned int posInt = id;
	d_intCos[posInt] = valC = d_paddingInImg[shiftImg];
	d_intSin[posInt] = valS = 0.0;



#pragma unroll
	for (int pos = 0; pos < nInt - 1; ++pos) {
		posInt = id + pos * nIntCos;
		posInt1 = posInt + nIntCos;

#ifdef ORIGINAL

		d_intCos[posInt1] = valM = __ldg(&d_cosL[loopOrd]) * valC - __ldg(&d_sinL[loopOrd]) * valS + d_paddingInImg[shiftImg + pos + 1];
		d_intSin[posInt1] = valS = __ldg(&d_sinL[loopOrd]) * valC + __ldg(&d_cosL[loopOrd]) * valS;

#else
#ifdef CHANGEPADDING
		d_intCos[posInt1] = valM = __ldg(&d_cosL[loopOrd]) * valC - __ldg(&d_sinL[loopOrd]) * valS + d_paddingInImg[shiftImg + pos + 1];
		d_intSin[posInt1] = valS = __ldg(&d_sinL[loopOrd]) * valC + __ldg(&d_cosL[loopOrd]) * valS;
#endif // CHANGEPADDING

#endif // ORIGINAL




		valC = valM;

	}
}

__global__ void gaussSmmothAfterIIR(float *d_blurImg, float *d_paddingInImg, float *d_intCos, float *d_blurCoef, int nOrd, int K, int nx, int ny) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int id = idx + idy * gridDim.x * blockDim.x;
	if (id > nx*ny)return;

	unsigned int K2 = 2 * K;
	unsigned int nInt = nx + K2;

	unsigned int iy = id / nx;
	unsigned int pos = iy * nInt * (nOrd + 1) + id % nx;



	float valB = 0.0;
#pragma unroll
	for (int loopOrd = 0; loopOrd <= nOrd; ++loopOrd) {
		valB += __ldg(&d_blurCoef[loopOrd]) * (d_intCos[pos + K2] - d_intCos[pos]);
		pos += nInt;
	}
	d_blurImg[id] = valB + __ldg(&d_blurCoef[nOrd + 1]) * d_paddingInImg[iy * nInt + id % nx];
}

__global__ void gaussDiffAfterIIR(float *d_diffImg, float *d_intSin, float *d_diffCoef, int nOrd, int K, int nx, int ny) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int id = idx + idy * gridDim.x * blockDim.x;
	if (id > nx*ny)return;

	unsigned int K2 = 2 * K;
	unsigned int nInt = nx + K2;

	unsigned int iy = id / nx;
	unsigned int pos = iy * nInt * (nOrd + 1) + id % nx;



	float valD = 0.0;
	for (int loopOrd = 0; loopOrd <= nOrd; ++loopOrd) {
		valD += __ldg(&d_diffCoef[loopOrd]) * (d_intSin[pos + K2] - d_intSin[pos]);
		pos += nInt;
	}
	d_diffImg[id] = valD;

}

__global__ void transposeSmem(float *d_out, float *d_in, int nrows, int ncols)
{
	// static shared memory
	__shared__ float tile[BDIMY][BDIMX];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = d_in[offset];
	}

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// NOTE - need to transpose row and col on block and thread-block level:
	// 1. swap blocks x-y
	// 2. swap thread x-y assignment (irow and icol calculations above)
	// note col still has continuous threadIdx.x -> coalesced gst
	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	  // NOTE nrows is stride of result, row and col are transposed
	unsigned int transposed_offset = INDEX(row, col, nrows);
	// thread synchronization
	__syncthreads();

	// NOTE invert sizes for write check
	if (row < ncols && col < nrows)
	{
		// store data to global memory from shared memory
		d_out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
	}
}

__global__ void paddingZeroGpu(float *d_outImg, float *d_inImg, const int nx, const int ny, const int K) {
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int idy = threadIdx.y + blockDim.y*blockIdx.y;
	unsigned int id = idx + idy * (blockDim.x*gridDim.x);
	unsigned int row = (nx + 2 * K);
	unsigned int amount = row * ny;
	if (id >= amount) return;

#ifdef ORIGINAL
	if (id % row >= K && id%row < (nx + K)) {

		unsigned int oldId = (id / row)*nx + (id%row) - K;
		d_outImg[id] = d_inImg[oldId];
	}
	else {
		d_outImg[id] = 0;
	}
#endif
}

__global__ void paddingNonzeroGpu(float *d_outImg, float *d_inImg, const int nx, const int ny, const int K) {
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int idy = threadIdx.y + blockDim.y*blockIdx.y;
	unsigned int id = idx + idy * (blockDim.x*gridDim.x);
#ifdef ORIGINAL

	unsigned int ncols = (nx + 2 * K);
	unsigned int amount = ncols * ny;

#else
#ifdef CHANGEPADDING

	unsigned int nrows = (ny + 2 * K);
	unsigned int amount = nrows * nx;

#endif // CHANGEPADDING
#endif // ORIGINAL

	if (id >= amount) return;
	unsigned int oldId;

#ifdef ORIGINAL
	if (id % ncols >= K && id%ncols < (nx + K)) {
		oldId = (id / ncols)*nx + (id%ncols) - K;
	}
	else if (id % ncols < K) {
		oldId = (id / ncols)*nx;
	}
	else {
		oldId = (id / ncols + 1) * nx - 1;
	}

#else
#ifdef CHANGEPADDING
	unsigned int newy = id / nx;
	if (newy >= K && newy < (ny + K)) {
		oldId = (newy - K)*nx + id % nx;
	}
	else if (newy < K) {
		oldId = id % nx;
	}
	else {
		oldId = (ny - 1)*nx + id % nx;
	}
#endif // OCHANGEPADDING

#endif // ORIGINAL
	d_outImg[id] = d_inImg[oldId];
}

/*padding img by gpu */
void paddingGpu(float *d_outImg, float *d_inImg, int nx, int ny, int K, int flag) {

	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + 2 * K + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


	switch (flag)
	{
	case 0:
		paddingZeroGpu << <grid, block >> > (d_outImg, d_inImg, nx, ny, K);
		break;
	case 1:
		paddingNonzeroGpu << <grid, block >> > (d_outImg, d_inImg, nx, ny, K);
	default:
		break;
	}

}

/*padding img by gpu asynchronizedly*/
void paddingGpuAsync(float *d_outImg, float *d_inImg, int nx, int ny, int K, int flag, cudaStream_t pStream) {

	dim3 block(BDIMX, BDIMY);
	dim3 grid((nx + 2 * K + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


	switch (flag)
	{
	case 0:
		paddingZeroGpu << <grid, block, 0, pStream >> > (d_outImg, d_inImg, nx, ny, K);
		break;
	case 1:
		paddingNonzeroGpu << <grid, block, 0, pStream >> > (d_outImg, d_inImg, nx, ny, K);
	default:
		break;
	}

}
