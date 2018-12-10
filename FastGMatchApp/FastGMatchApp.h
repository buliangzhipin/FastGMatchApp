#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_FastGMatchApp.h"
#include "MatchThread.h"
#include "MakeThread.h"
#include <QDebug>
#include <QtGui>
#include "kernel.cuh"
#include "utilities.h"
#include "mainProcessGPU.h"
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "GraphicsProcess.h"
#include "GraphcisChangeView.h"

using namespace cv;

class FastGMatchApp : public QMainWindow
{
	Q_OBJECT

public:
	FastGMatchApp(QWidget *parent = Q_NULLPTR);

signals:

private slots:
	void startGMatchGPU();
	void startGMakeGPU();
	void changeScaleValue(int value);
	void loadImage();
	void dealErrorMessage(QString title,QString message);
	void loadChangedImage();

private:
	Ui::FastGMatchAppClass *ui = new Ui::FastGMatchAppClass();
	MatchThread *mtThread;
	MakeThread *mkThread;
	QThread *qmtThread[2];

private:
	int scaleTmp;
	int scaleMaxTmp;
	float makeScaleRatio = 1.414;
	int scaleObjectMaxNumber = 4;
	QSlider *sliderGroup[4];
	QLCDNumber *lcdGroup[4];
	int scaleGroup[4];
	QList<int> makeScaleList;

private:
	void resizeEvent(QResizeEvent *event);
	//void paintEvent(QPaintEvent *event);
	Mat imageBuffer;
	QPixmap imageBufferPix;
	QList<Mat> qlistMat;
	GraphcisChangeView *gcv;
};
