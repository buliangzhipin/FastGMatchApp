#pragma once
#include <qlist.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

typedef struct {
	int nTran;
	float *gptTbl;
} GptTbl; /* Work max for theta*/

class GraphicsProcess
{
public:
	GraphicsProcess();
	~GraphicsProcess();
	QList<cv::Mat> changeGraph(cv::Mat inImg);
	int getNTran();
private:
	GptTbl* gptTbl = new GptTbl();
};

