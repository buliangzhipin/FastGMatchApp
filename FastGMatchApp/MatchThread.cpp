#if defined(_MSC_VER) && (_MSC_VER >= 1600)

# pragma execution_character_set("utf-8")

#endif

#include "MatchThread.h"
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "qmessagebox.h"

#include "mainProcessGPU.h"
#include "utilities.h"


using namespace cv;

MatchThread::MatchThread(QObject *parent)
	: QObject(parent)
{
}

MatchThread::~MatchThread()
{
}


void MatchThread::matchImage() {
	QByteArray str_arr = fileName.toLocal8Bit();
	const char* c_str = str_arr.constData();

	VideoCapture capture;
	Mat camera;
	Mat gray;
	Mat grayF;
	int *data = new int[10];
	//采用 Directshow 的方式打开第一个摄像头设备。
	capture.open(0, CAP_DSHOW);
	if (!capture.isOpened())
	{
		delete[] data;
		emit errorMessage("Camera Error", "カメラがみつからない");
		return;
	}
	//capture.set(CAP_PROP_SETTINGS,0);//调出 Directshow 摄像头属性设置栏
	capture.read(camera);
	MainProcessGPU mpg(camera.cols, camera.rows, scale, scaleMax, 2.0);
	try
	{
		mpg.loadFeature(c_str);
		while (true)
		{
			//读取一帧图像
			capture.read(camera);
			cvtColor(camera, gray, COLOR_BGR2GRAY);
			gray.convertTo(grayF, CV_32F);
			mpg.calPoint((float*)grayF.data, data);
			if (camera.empty())
			{
				continue;
			}

			Point p(data[0], data[1]);
			circle(camera, p, data[2], Scalar(0, 0, 255), 3);
			imshow("Matching", camera);

			//qDebug() << data[0] << "   " << data[1] ;

			//Esc
			if (waitKey(1) == 27 || !cvGetWindowHandle("Matching"))
			{
				break;
			}
		}
	}
	catch (FileLoadError f)
	{
		emit errorMessage("File Error", "Featureファイルが存在しない");
		//QMessageBox::about(this, "File Error", "Featureファイルが存在しない");
	}

	delete[] data;
	cv::destroyAllWindows();
	capture.release();

}
