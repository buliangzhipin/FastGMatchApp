//#if defined(_MSC_VER) && (_MSC_VER >= 1600)
//
//# pragma execution_character_set("utf-8")
//
//#endif

#include "MakeThread.h"



MakeThread::MakeThread(QObject *parent)
	: QObject(parent)
{
}

MakeThread::~MakeThread()
{
}

void MakeThread::makeImage() {
	QByteArray str_arr = fileName.toLocal8Bit();
	const char* c_str = str_arr.constData();

	cv::Mat grayF;
	grayimg->convertTo(grayF, CV_32F);
	MainProcessGPU mpg(grayimg->cols, grayimg->rows, scale, scaleMax, 1.414);
	mpg.savePoint((float*)grayF.data, c_str);

}