#pragma once

#include <QObject>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "qmessagebox.h"
#include "mainProcessGPU.h"



class MakeThread : public QObject
{
	Q_OBJECT

public:
	MakeThread(QObject *parent = nullptr);
	~MakeThread();
	QString fileName;
	int scale;
	int scaleMax;
	cv::Mat *grayimg;
	QList<cv::Mat> grayImgList;


public slots:
	void makeImage();

signals:
	void errorMessage(QString title, QString message);
};
