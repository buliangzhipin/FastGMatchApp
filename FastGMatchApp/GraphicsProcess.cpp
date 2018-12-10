#include "GraphicsProcess.h"
#include <math.h>
#include <utilities.h>
using namespace std;


#define PI             3.141592653589793
/* To make template */
#define NTEMPLATEROT       4       /* Number of types of rotation for template image */
//#define NTEMPLATEROT      128
#define NTEMPLATEENLONG    2       /* Number of types of anisotropic elongation for template image */
#define TEMPLATEENCOEF  0.25       /* The coefficient for anisotropic elongation */

GraphicsProcess::GraphicsProcess()
{
	int iEnlong;
	float cosTran, sinTran, alpha, beta;
	gptTbl->nTran = NTEMPLATEROT * NTEMPLATEENLONG + 1;
	gptTbl->gptTbl = new float[9 * gptTbl->nTran];
	gptTbl->gptTbl[C11] = gptTbl->gptTbl[C22] = 1.0;
	gptTbl->gptTbl[C12] = gptTbl->gptTbl[C21] = 0.0;
	gptTbl->gptTbl[C11] = gptTbl->gptTbl[C22] = 1.0;
	gptTbl->gptTbl[C13] = gptTbl->gptTbl[C23] = gptTbl->gptTbl[C31] = gptTbl->gptTbl[C32] = 0.0;
	gptTbl->gptTbl[C33] = 1.0;
	int pos = 0,iTheta;
	for (iTheta = 0; iTheta < NTEMPLATEROT; ++iTheta) {
		cosTran = cos((PI / NTEMPLATEROT) * iTheta); 		sinTran = sin((PI / NTEMPLATEROT) * iTheta);
		for (iEnlong = 1; iEnlong <= NTEMPLATEENLONG; ++iEnlong) {
			pos += 9;
			alpha = 1.0 + TEMPLATEENCOEF * iEnlong; beta = 1.0 - TEMPLATEENCOEF * iEnlong;
			//printf("alpha %f  beta %f\n", alpha, beta);
			gptTbl->gptTbl[C11 + pos] = alpha * cosTran * cosTran + beta * sinTran * sinTran;
			gptTbl->gptTbl[C22 + pos] = beta * cosTran * cosTran + alpha * sinTran * sinTran;
			gptTbl->gptTbl[C12 + pos] = gptTbl->gptTbl[C21 + pos] = (alpha - beta) * cosTran * sinTran;
			gptTbl->gptTbl[C13 + pos] = gptTbl->gptTbl[C23 + pos] = gptTbl->gptTbl[C31 + pos] = gptTbl->gptTbl[C32 + pos] = 0.0;
			gptTbl->gptTbl[C33 + pos] = 1.0;
			//gptPr(&(gptTbl[pos]), "GPTInit ");
		}
	}
}


GraphicsProcess::~GraphicsProcess()
{
}

QList<cv::Mat> GraphicsProcess::changeGraph(cv::Mat inImgOrg)
{
	cv::Mat gray;
	cvtColor(inImgOrg, gray, cv::COLOR_BGR2GRAY);
	cv::Mat grayF;
	gray.convertTo(grayF, CV_32F);

	QList<cv::Mat> matList;
	for (int i = 0; i < gptTbl->nTran; i++) {
		float *inImg = new float[grayF.cols*grayF.rows];
		gptTransformImage(&(gptTbl->gptTbl[i * 9]), (float*)grayF.data, inImg, grayF.cols, grayF.rows, grayF.cols /2, grayF.rows /2);
		cv::Mat inImgMat(grayF.rows, grayF.cols, CV_32FC1, inImg);
		cv::Mat *outImgMat = new cv::Mat();
		inImgMat.convertTo(*outImgMat, CV_8UC1);
		matList.append(*outImgMat);
		delete inImg;
	}

	return matList;
}

int GraphicsProcess::getNTran()
{
	return gptTbl->nTran;
}
