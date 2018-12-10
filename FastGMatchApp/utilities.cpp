/*
 * utilities.c
 *
 *  Created on: 2018/05/01
 *      Author: yamasita
 */
#include "kernel.cuh"
#include "utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define EPS 1.0e-10

int gptInverse(float *gpt, float *iGpt);
void gptTransformPoint(float *gpt, float *inP, float *outP);
void mkHeap(float *record, int *index, int leaf, int root);

/* Heap sort */
void heapSort(float *record, int *index, int nData) {
	int    root, leaf, tmpIndex;
	float tmp;
	leaf = nData - 1;      /* Leaf (position is the index of array) */
	root = nData / 2 - 1;  /* Root*/

	/* Initial semi-ordered tree construction */
	while (root >= 0) {
		mkHeap(record, index, leaf, root);
		--root;
	}
	/* Extract the smallest element from semi-ordered tree */
	while (leaf > 0) {
		/* Exchange root and leaf */
		tmp = record[0];
		record[0] = record[leaf];
		record[leaf] = tmp;
		tmpIndex = index[0];
		index[0] = index[leaf];
		index[leaf] = tmpIndex;
		/* Reconstruct semi-ordered tree */
		--leaf;
		mkHeap(record, index, leaf, 0);
	}
}

/* Make semi-ordered tree when root is not correct position for heap sort*/
void mkHeap(float *record, int *index, int leaf, int root) {
	int min, tmpIndex;
	int child = root * 2 + 1;
	float tmp;

	while (child <= leaf) {
		if (child < leaf && record[child + 1] < record[child]) {
			min = child + 1;
		}
		else {
			min = child;
		}
		if (record[root] <= record[min]) break;
		tmp = record[root];
		record[root] = record[min];
		record[min] = tmp;
		tmpIndex = index[root];
		index[root] = index[min];
		index[min] = tmpIndex;
		root = min;
		child = root * 2 + 1;
	}
}

/* Load image (Input of header & body information of pgm file) */
void loadImageFile(char *fileName, int *img, int nx, int ny) {
	/* unsigned char buffer[MAX_BUFFERSIZE];*/
	char buffer[MAX_BUFFERSIZE];
	FILE *fp;         /* File pointer */
	int ntx, nty;     /* Original image size */
	int max_gray;    /* Maximum gray level */
	int pos, ix, iy; /* Loop variable */

	/* Input file open */
	//printf("Input file = %s\n", fileName);
	fp = fopen(fileName, "rb");
	if (fp == NULL) {
		printf("     The file doesn't exist! %s \n\n", fileName);
		exit(1);
	}

	/* Check of file-type ---P5 */
	if (fgets(buffer, MAX_BUFFERSIZE, fp) == 0) {
		printf("     Read error of image file %s \n\n", fileName);
		exit(1);
	}
	if (buffer[0] != 'P' || buffer[1] != '5') {
		printf("     Mistaken file format, not P5! %s \n\n", fileName);
		exit(1);
	}

	/* input of x_size1, y_size1 */
	ntx = 0; nty = 0;
	while (ntx == 0 || nty == 0) {
		if (fgets(buffer, MAX_BUFFERSIZE, fp) == 0) {
			printf("     Read error of image file %s \n\n", fileName);
			exit(1);
		}
		if (buffer[0] != '#')  sscanf(buffer, "%d %d", &ntx, &nty);
	}

	if (ntx != nx || nty != ny) {
		printf("     Image size is not correct ! %s  (%d, %d) != (%d, %d) \n\n", fileName, ntx, nty, nx, ny);
		exit(1);
	}

	/* input of max_gray */
	max_gray = 0;
	while (max_gray == 0) {
		if (fgets(buffer, MAX_BUFFERSIZE, fp) == 0) {
			printf("     Read error of image file %s \n\n", fileName);
			exit(1);
		}
		if (buffer[0] != '#')  sscanf(buffer, "%d", &max_gray);
	}

	if (max_gray != WHITE) {
		printf("     Invalid value of maximum gray level! %s \n\n", fileName);
		exit(1);
	}

	/* Input of image data */
	pos = 0;
	for (iy = 0; iy < nty; ++iy) {
		for (ix = 0; ix < ntx; ++ix) {
			img[pos++] = (int)((unsigned char)fgetc(fp));
		}
	}
	fclose(fp);
	//#define SIMPLEIMAGE
#ifdef SIMPLEIMAGE
	for (iy = 0; iy < ny; ++iy) {
		for (ix = 0; ix < nx; ++ix) {
			//		if (ix % 64 <32)  	img[ix + 2 * nx * iy] = 1; else img[ix + 2 * nx * iy] = 0;
			//			if ((iy - 255) > 0 && (ix - 255) > 0) 	img[ix + nx * iy] = 1.0; else img[ix + 2 * nx * iy] = 0;/* Small square */
			if (ix * 2 >= ny)  	img[ix + nx * iy] = 1; else img[ix + nx * iy] = 0;
		}
	}
#endif
}

/* Save image (FP). Values are normalized */
void saveImageFileFp(char *fileName, float *img, int nx, int ny) {
	FILE *fp; /* File pointer */
	int  pos; /* Loop variable */
	float maxVal, minVal, coef;

	fp = fopen(fileName, "wb");
	if (fp == NULL) {
		printf("     Cannot open! %s \n\n", fileName);
		exit(1);
	}
	/* output of pgm file header information */
	fputs("P5\n", fp);
	fputs("# Created by Image Processing\n", fp);
	fprintf(fp, "%d %d\n", nx, ny);
	fprintf(fp, "%d\n", WHITE);

	/* Output of image data */
	maxVal = minVal = img[0];
	for (pos = 1; pos < nx * ny; ++pos) {
		if (maxVal < img[pos]) maxVal = img[pos];
		else if (minVal > img[pos]) minVal = img[pos];
	}
	coef = (WHITE - BLACK) / (maxVal - minVal);
	//printf("%f  %f \n", maxVal, minVal);
	for (pos = 0; pos < nx * ny; ++pos) {
		fputc((int)(coef * (img[pos] - minVal)), fp);
	}
	fclose(fp);
}

/* Save image (int) */
void saveImageFile(char *fileName, int *img, int nx, int ny) {
	FILE *fp; /* File pointer */
	int  pos; /* Loop variable */

	fp = fopen(fileName, "wb");
	if (fp == NULL) {
		printf("     Cannot open! %s \n\n", fileName);
		exit(1);
	}
	/* output of pgm file header information */
	fputs("P5\n", fp);
	fputs("# Created by Image Processing\n", fp);
	fprintf(fp, "%d %d\n", nx, ny);
	fprintf(fp, "%d\n", WHITE);

	/* Output of image data */
	for (pos = 0; pos < nx * ny; ++pos) {
		if (img[pos] > WHITE) fputc(WHITE, fp);
		else if (img[pos] < BLACK) fputc(BLACK, fp);
		else fputc(img[pos], fp);
	}
	fclose(fp);
}

/* Print information of features */
void prFeatureInf(Feature *feature) {
	printf("Feature Info. (%3d, %3d) order = %2d, angle = %2d, scale = %7.2f\n", feature->ix, feature->iy, feature->ordHist, feature->iTheta, feature->scale);
}
/* Load features */
void loadFeatures(const char *fileName, Features *features) {
	int  pos, ret = 0;
	Feature *feature;
	FILE *fp; /* File pointer */
	fp = fopen(fileName, "rb");
	if (fp == NULL) {
		throw FileNotFound;
	}

	ret += fread(&(features->nx), sizeof(int), 1, fp);
	ret += fread(&(features->ny), sizeof(int), 1, fp);
	ret += fread(&(features->nFeature), sizeof(int), 1, fp);
	ret += fread(&(features->nAngleCoef), sizeof(int), 1, fp);

	features->nEstFeature = features->nFeature;
	features->featureL = (Feature *)malloc(sizeof(Feature) * features->nEstFeature);
	for (int pos = 0; pos < features->nEstFeature; ++pos) {
		features->featureL[pos].vector = (float *)malloc(sizeof(float) * (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE);
	}

	for (pos = 0; pos < features->nFeature; ++pos) {
		feature = &(features->featureL[pos]);
		ret += fread(&(feature->ix), sizeof(int), 1, fp);
		ret += fread(&(feature->iy), sizeof(int), 1, fp);
		ret += fread(&(feature->ordHist), sizeof(int), 1, fp);
		ret += fread(&(feature->iTheta), sizeof(int), 1, fp);
		ret += fread(&(feature->scale), sizeof(float), 1, fp);
		ret += fread(&(feature->transType), sizeof(int), 1, fp);
		ret += fread(feature->vector, sizeof(float), (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE, fp);
	}
	if (ret != 0) {
		printf("Error occurs !\n");
	}
}

/* Save features */
void saveFeatures(const char *fileName, Features *features) {
	int  pos;
	Feature *feature;
	FILE *fp; /* File pointer */
	fp = fopen(fileName, "wb");
	if (fp == NULL) {
		throw FileNotFound;
	}

	fwrite(&(features->nx), sizeof(int), 1, fp);
	fwrite(&(features->ny), sizeof(int), 1, fp);
	fwrite(&(features->nFeature), sizeof(int), 1, fp);
	fwrite(&(features->nAngleCoef), sizeof(int), 1, fp);

	for (pos = 0; pos < features->nFeature; ++pos) {
		feature = &(features->featureL[pos]);
		fwrite(&(feature->ix), sizeof(int), 1, fp);
		fwrite(&(feature->iy), sizeof(int), 1, fp);
		fwrite(&(feature->ordHist), sizeof(int), 1, fp);
		fwrite(&(feature->iTheta), sizeof(int), 1, fp);
		fwrite(&(feature->scale), sizeof(float), 1, fp);
		fwrite(&(feature->transType), sizeof(int), 1, fp);
		fwrite(feature->vector, sizeof(float), (2 * PANGLE + 1) * GRIDSIZE * GRIDSIZE, fp);
	}
	fclose(fp);
}

/* Inverse projection transformation */
int gptInverse(float *gpt, float *iGpt) {
	float det;
	det = gpt[C11] * gpt[C22] * gpt[C33]
		+ gpt[C21] * gpt[C32] * gpt[C13]
		+ gpt[C31] * gpt[C12] * gpt[C23]
		- gpt[C11] * gpt[C32] * gpt[C23]
		- gpt[C21] * gpt[C12] * gpt[C33]
		- gpt[C31] * gpt[C22] * gpt[C13];
	if (fabs(det) < EPS) {
		printf("Singular GPT is generated!!! in gptInverse() \n");
		return 1;
	}
	iGpt[C11] = (gpt[C22] * gpt[C33] - gpt[C32] * gpt[C23]) / det;
	iGpt[C21] = -(gpt[C21] * gpt[C33] - gpt[C31] * gpt[C23]) / det;
	iGpt[C31] = (gpt[C21] * gpt[C32] - gpt[C31] * gpt[C22]) / det;

	iGpt[C12] = -(gpt[C12] * gpt[C33] - gpt[C32] * gpt[C13]) / det;
	iGpt[C22] = (gpt[C11] * gpt[C33] - gpt[C31] * gpt[C13]) / det;
	iGpt[C32] = -(gpt[C11] * gpt[C32] - gpt[C31] * gpt[C12]) / det;

	iGpt[C13] = (gpt[C12] * gpt[C23] - gpt[C22] * gpt[C13]) / det;
	iGpt[C23] = -(gpt[C11] * gpt[C23] - gpt[C21] * gpt[C13]) / det;
	iGpt[C33] = (gpt[C11] * gpt[C22] - gpt[C21] * gpt[C12]) / det;
	return 0;
}

void gptTransformPoint(float *gpt, float *inP, float *outP) {
	int i, j;
	float sum;
	for (i = 0; i < 3; ++i) {
		sum = 0.0;
		for (j = 0; j < 3; ++j) {
			sum += gpt[i + j * 3] * inP[j];
		}
		outP[i] = sum;
	}
}

/* Projection transformation of the image by bilinear interpolation */
void gptTransformImage(float *inGpt, float *inImg, float *outImg, int nx, int ny, int cx, int cy) {
	int x, y, xOrg, yOrg, posOrg;
	float xOrgFl, xOrgFrac, yOrgFl, yOrgFrac;
	float inVect[3], outVect[3], gpt[9];

	/* Inverse transform of GPT */
	gptInverse(inGpt, gpt);
	//gptPr(gpt, "GPT ");
	/* Output image generation by bilinear interpolation */
	inVect[2] = 1.0;
	for (y = 0; y < ny; y++) {
		inVect[1] = y - cy;
		for (x = 0; x < nx; x++) {
			inVect[0] = x - cx;
			gptTransformPoint(gpt, inVect, outVect);
			xOrgFl = outVect[0] / outVect[2] + cx;
			yOrgFl = outVect[1] / outVect[2] + cy;
			xOrg = (int)floor(xOrgFl);
			yOrg = (int)floor(yOrgFl);
			xOrgFrac = xOrgFl - xOrg;
			yOrgFrac = yOrgFl - yOrg;
			if (xOrg >= 0 && xOrg + 1 < nx && yOrg >= 0 && yOrg + 1 < ny) {
				posOrg = yOrg * nx + xOrg;
				outImg[y * nx + x] = (
					(1.0 - yOrgFrac) * ((1.0 - xOrgFrac) * inImg[posOrg] + xOrgFrac * inImg[posOrg + 1])
					+ yOrgFrac * ((1.0 - xOrgFrac) * inImg[posOrg + nx] + xOrgFrac * inImg[posOrg + nx + 1]));
			}
			else {
				outImg[y * nx + x] = 0.0;
			}
		}
	}
}

/* Show GPT parameter */
void gptPr(float *gpt, char *st) {
	int i;
	printf("%s \n", st);
	for (i = 0; i < 3; ++i) {
		printf("%10.6f  %10.6f  %10.6f\n", gpt[i], gpt[i + 3], gpt[i + 6]);
	}
}

/* Rotate image (This can be done by gptTransformImage() )*/
int rotImg(float *inImg, float *outImg, float theta, int nx, int ny, int nxOut, int nyOut) {
	int cx = nx / 2, cy = ny / 2;
	int cxOut = nxOut / 2, cyOut = nyOut / 2;
	int ixOut, iyOut;
	int itx, ity, pos;
	float tx, ty, fx, fy;
	float cost = cos(theta), sint = sin(theta);


	for (iyOut = 0; iyOut < nyOut; ++iyOut) {
		for (ixOut = 0; ixOut < nxOut; ++ixOut) {
			tx = cost * (ixOut - cxOut) + sint * (iyOut - cyOut) + cx;
			ty = -sint * (ixOut - cxOut) + cost * (iyOut - cyOut) + cy;
			itx = (int)tx;
			ity = (int)ty;
			fx = tx - itx; fy = ty - ity;
			pos = itx + nx * ity;
			outImg[ixOut + nxOut * iyOut] = (1.0 - fx) * (1.0 - fy) * inImg[pos] + fx * (1.0 - fy) * inImg[pos + 1]
				+ (1.0 - fx) * fy * inImg[pos + nx] + fx * fy * inImg[pos + nx + 1];
		}
	}
	return 0;
}

/* Save edge histogram as images*/
//int histSave(float *dirHist, int nx, int ny) {
//	int nxy = nx * ny;
//	char fileName[MAX_FILENAME];
//
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "histc0.pgm");
//	saveImageFileFp(fileName, dirHist, nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "histc1.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "histc2.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 3]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "histc3.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 5]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "histc4.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 7]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "hists1.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 2]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "hists2.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 4]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "hists3.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 6]), nx, ny);
//	strcpy(fileName, WORKOUTDIR); strcat(fileName, "hists4.pgm");
//	saveImageFileFp(fileName, &(dirHist[nxy * 8]), nx, ny);
//
//	return 0;
//}


/* save FP data as text */
void saveData(char *fileName, float *data, int nData) {
	FILE *fp; /* File pointer */
	int  pos; /* Loop variable */

	fp = fopen(fileName, "w");
	if (fp == NULL) {
		printf("     Cannot open! %s \n\n", fileName);
		exit(1);
	}
	/* Output of image data */
	for (pos = 0; pos < nData; ++pos) {
		fprintf(fp, "%d  %e\n", pos, data[pos]);
	}
	fclose(fp);
}

