#pragma once

#ifndef FASTGMATCH_H_
#define FASTGMATCH_H_

/* Default parameters: switch whether making template or matching, file names. They can be changed by command line parameters */
#define MKTEMPLATE     0               /* 0: matching, 1: make template */
//#define TEMPLATEDNAME "img1.template"  /* Template data file name. It used for both input (matching) and output (making template) */
//#define IMAGENAME     "img1"           /* Input image file name */

/* Floading point type (float or double) */
#define FPTYPE float

/* To make template */
#define NTEMPLATEROT       4       /* Number of types of rotation for template image */
//#define NTEMPLATEROT      128
#define NTEMPLATEENLONG    2       /* Number of types of anisotropic elongation for template image */
#define TEMPLATEENCOEF  0.25       /* The coefficient for anisotropic elongation */
#define TEMPLSCALERATIO 1.41421356 /* the next scale = the present scale * SCALERATIO */

///* Selection of image database */
//#define GRAF  /* BOAT, GRAF, or LENNA*/
//
///* Image information */
//#ifdef GRAF
//#define WORKDIR         "../images/GRAF/"
//#define WORKOUTDIR      "../images/GRAF/OutMatch/"
//#define TEMPLATEX       430
//#define TEMPLATEY       310
//#define TEMPLSCALEINIT  150.0     /* The initial scale */
//#define TEMPLSCALEMAX   250.0     /* The maximum scale to be handled */
//#define NX              800      /* Horizontal # of pixel in image */
//#define NY              640      /* Vertical # of pixel in image */
//#define SCALEINIT       80.0     /* The initial scale */
//#define SCALEMAX        161.0    /* The maximum scale to be handled */
//#endif
//
//#ifdef BOAT
//#define WORKDIR         "../images/Boat/"
//#define WORKOUTDIR      "../images/Boat/OutMatch/"
//#define TEMPLATEX       425
//#define TEMPLATEY       340
//#define TEMPLSCALEINIT  200.0     /* The initial scale */
//#define TEMPLSCALEMAX   300.0     /* The maximum scale to be handled */
//#define NX              850      /* Horizontal # of pixel in image */
//#define NY              680      /* Vertical # of pixel in image */
//#define SCALEINIT       75.0     /* The initial scale */
//#define SCALEMAX        301.0    /* The maximum scale to be handled */
//#endif
//
//#ifdef LENNA  /* This file is used only for debug (Not matching) */
//#define WORKDIR         "../images/Lenna/"
//#define WORKOUTDIR      "../images/Lenna/OutMatch/"
//#define TEMPLATEX       256
//#define TEMPLATEY       256
//#define TEMPLSCALEINIT   80.0     /* The initial scale */
//#define TEMPLSCALEMAX    81.0     /* The maximum scale to be handled */
//#define NX              512      /* Horizontal # of pixel in image */
//#define NY              512      /* Vertical # of pixel in image */
//#define SCALEINIT        80.0     /* The initial scale for */
//#define SCALEMAX         81.0    /* The maximum scale to be handled */
//#endif

/* Parameters for calculating feature */
#define PGRAD      4        /* P to calculate of gradient and blur*/
#define PBLUR      2        /* P to calculate  of only blur */
#define PANGLE     4        /* P to express edge angle */
#define SIGMA2     1.1      /* sigma when P = 2 */
#define SIGMA4     0.80     /* sigma when P = 4 */
#define MAXK       600     /* The maximum K (to allocate memory) */
#define SCALERATIO 2.0      /* The next scale = the present scale * SCALERATIO */
#define GRADRATIO  0.5      /* Ratio for blur of gradient comparing to simple blur */
#define STEPRATIO  0.2      /* The sample points is scale * STEPRATIO */
#define GRIDSIZE   4        /* The grid size for feature vector = GRIDSIZE x GRIDSIZE */
#define NMAXTHETA  16       /* Number for thetas to search the maximum point of histogram */
//#define NMAXTHETA  512
#define HISTREG              /* Use regularization for edge histogram */
#define EDGEPOWER  1.0e-12   /* Regularization parameter or minimum power of gradient */

/* Calculate large scale or approximate from small scale */
#define USELARGESCALE  /* Define Calculate large scale. Not define: Approximate from small scale*/
/* Parameter for large scale approximation */
#define NAPPROPOINT     (N0AP + N1AP + N2AP + N3AP)  /* 1 + 8 + 16 + 24 = 49*/
#define N0AP   1
#define N1AP   8
#define N2AP   16
#define N3AP   24
#define R0AP   0.0
#define R1AP   0.6
#define R2AP   1.2
#define R3AP   1.8
#define A0AP   0.73975
#define A1AP   0.59329
#define A2AP   0.35835
#define A3AP   0.14622

/* Fixed values */
#define WHITE          255
#define BLACK          0
#define MAX_BUFFERSIZE 256                /* Buffer size to read text from a file */
#define MAX_FILENAME   256                /* Filename length limit  */
#define PI             3.141592653589793
#define ROUNDFRAC      0.49999            /* For round */

#endif /* FASTGMATCH_H_ */

#ifndef CALFEATURE_H_
#define CALFEATURE_H_
#include "cuda_runtime.h"

typedef struct Feature {
	int ix, iy;    /* Position */
	int ordHist;   /* 0 or 1 */
	int iTheta;
	FPTYPE scale;  /* Scale */
	int transType; /* Transformation type */
	FPTYPE *vector;
} Feature;

typedef struct {
	int nx, ny, nxy;
	int nFeature;
	int nAngleCoef;
	int nEstFeature;
	FPTYPE  *relativePosL;
	int     *relativePosScL;
	Feature *featureL;
	FPTYPE  *tmpVector;
} Features;

typedef struct {
	int K;
	int nInt;
	int nOrd;
	int maxNInt;
	int extType; /* 0 : zero extension, 1 : extension */
	FPTYPE *cosL;
	FPTYPE *sinL;
	FPTYPE *lineExt;
	FPTYPE *intSin;
	FPTYPE *intCos;
	FPTYPE *blurCoef;
	FPTYPE *diffCoef;
} WorkIIR;

typedef struct {
	int nx, ny;
	FPTYPE *xBlurImg;
	FPTYPE *xDiffXImg;
	FPTYPE *trans0;
	FPTYPE *trans1;
	FPTYPE *trans2;
} WorkImg;

typedef struct {
	int nx, ny;
	FPTYPE *invFourTbl;
	FPTYPE *rotTbl;
	FPTYPE *largeScaleRelTbl;
	int    *largeScaleTbl;
	FPTYPE *largeScaleATbl;
} WorkMT; /* Work max for theta*/




typedef struct {
	int K;
	int nInt;
	int nOrd;
	int maxNInt;
	int extType; /* 0 : zero extension, 1 : extension */
	FPTYPE *d_cosL;//kによる
	FPTYPE *d_sinL;//kによる
	FPTYPE *d_intSin;
	FPTYPE *d_intCos;
	FPTYPE *d_intSinDiff;
	FPTYPE *d_intCosDiff;
	FPTYPE *d_blurCoef;//定数
	FPTYPE *d_diffCoef;//定数
	FPTYPE *d_inImg;
	FPTYPE *d_paddingImg;
	FPTYPE *d_paddingImgdiff;
	FPTYPE *d_blurImg; //also diffx
	FPTYPE *d_diffImg; //also diffy
	FPTYPE *d_tranblurImg;
	FPTYPE *d_trandiffImg;
	FPTYPE *d_tranIIRCos;
	FPTYPE *d_tranIIRSin;
}WorkIIRGPU;/*iirをGPUで使うとき使用する部分
Kは変動する
sinLとcosLはkによって変動する
intCosとintSinはGPU上だけ計算する
diffはdiffYで使う部分です
*/


int maxDirection(FPTYPE *blurLargeDirHist, WorkMT *workMT, int *maxITheta);
int rotateDirHist(FPTYPE *vectorIn, FPTYPE *vectorOut, int stRot, WorkMT *workMT);
int mkFeature(FPTYPE *blurSmallDirHist, WorkMT *workMT, Features *features, Feature *feature);
int approxLargeScale(FPTYPE *blurSmallDirHist, FPTYPE *outHist, int ix, int iy, WorkMT *workMt);


void initialGPU4(WorkIIRGPU *workIIR4, int nx, int ny, int nOrd,int maxK);
void deleteGPU4(WorkIIRGPU *workIIR4);
void initialGPU2(WorkIIRGPU *workIIR2, int nx, int ny, int nOrd,int maxK);
void deleteGPU2(WorkIIRGPU *workIIR2);

int calDirHistPointGPU(FPTYPE *d_diffXImg, FPTYPE *d_diffYImg, FPTYPE *d_dirHist, FPTYPE *dirHist, int nx, int ny);
int calDirHistPointGPUAsync(FPTYPE *d_diffXImg, FPTYPE *d_diffYImg, FPTYPE *d_dirHist, int nx, int ny);
void gaussSmoothGPUAsync(float *blurImg, float *inImg, WorkIIRGPU *workiir, int nx, int ny, int K, cudaStream_t pStream);
void gaussSmoothGPUAsyncWidth(float *blurImg, float *blurImg2, float*inImg, WorkIIRGPU **workiir, int nx, int ny, int K, int K2, cudaStream_t *pStream, int times);
void gaussSmoothGPUAsyncHeight(float *blurImg, float *blurImg2, float*inImg, WorkIIRGPU **workiir, int nx, int ny, int K, int K2, cudaStream_t *pStream, int times);
void gaussDiffGPUAsync(float *diffXImg, float *diffYImg, float *inImg, WorkIIRGPU *workiir, int nx, int ny, int K, cudaStream_t *pStream);

#include "qdebug.h"
#include "utilities.h"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        qDebug() << "code: "<< error<<", reason: "<<  cudaGetErrorString(error);                      \
        throw FileNotFound;                                                               \
    }                                                                          \
}
//fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \

#endif /* CALFEATURE_H_ */

