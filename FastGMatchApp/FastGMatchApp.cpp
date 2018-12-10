#if defined(_MSC_VER) && (_MSC_VER >= 1600)

# pragma execution_character_set("utf-8")

#endif

#include "FastGMatchApp.h"



#include <qfiledialog.h>
#include <qmessagebox.h>
#include "utilities.h"
#include <qthread.h>


using namespace cv;


FastGMatchApp::FastGMatchApp(QWidget *parent)
	: QMainWindow(parent)
{
	ui->setupUi(this);

	//load image
	connect(ui->loadImageMake, &QPushButton::clicked, this, &FastGMatchApp::loadImage);
	connect(ui->imageButton, &QPushButton::clicked, this, &FastGMatchApp::loadChangedImage);
	gcv =new GraphcisChangeView(this);

	//scale change process
	sliderGroup[0] = ui->scaleSliderMake;
	sliderGroup[1] = ui->maxscaleSliderMake;
	sliderGroup[2] = ui->scaleSliderMatch;
	sliderGroup[3] = ui->maxscaleSliderMatch;

	lcdGroup[0] = ui->scaleLcdMake;
	lcdGroup[1] = ui->maxscaleLcdMake;
	lcdGroup[2] = ui->scaleLcdMatch;
	lcdGroup[3] = ui->maxscaleLcdMatch;

	for (int i = 0; i < scaleObjectMaxNumber; i++) {
		scaleGroup[i] = sliderGroup[i]->value();
		lcdGroup[i]->display(scaleGroup[i]);
		connect(sliderGroup[i], &QSlider::valueChanged, this, &FastGMatchApp::changeScaleValue);
	}


	//gpu match process
	qmtThread[0] = new QThread(this); 
	qmtThread[1] = new QThread(this);
	mtThread = new MatchThread();
	mtThread->moveToThread(qmtThread[0]);
	mkThread = new MakeThread();
	mkThread->moveToThread(qmtThread[1]);
	connect(qmtThread[0], &QThread::started, mtThread, &MatchThread::matchImage);
	connect(mtThread, &MatchThread::errorMessage, this, &FastGMatchApp::dealErrorMessage);
	connect(ui->gpuStartButton, &QPushButton::clicked, this, &FastGMatchApp::startGMatchGPU);
	connect(qmtThread[1], &QThread::started, mkThread, &MakeThread::makeImage);
	connect(mkThread, &MakeThread::errorMessage, this, &FastGMatchApp::dealErrorMessage);
	connect(ui->gpuMakeStartButton, &QPushButton::clicked, this, &FastGMatchApp::startGMakeGPU);
}

void FastGMatchApp::changeScaleValue(int value) {
	QSlider *senderWidget = (QSlider *)sender();
	int objectNumber;
	for (objectNumber = 0; objectNumber < scaleObjectMaxNumber;objectNumber++) {
		if (senderWidget == sliderGroup[objectNumber])break;
	}
	if (objectNumber % 2 == 0) {
		scaleGroup[objectNumber] = senderWidget->value();
		if (scaleGroup[objectNumber]>sliderGroup[objectNumber+1]->value()) {
			sliderGroup[objectNumber + 1]->setValue(senderWidget->value() + 1);
		}	
	}
	else {
		scaleGroup[objectNumber] = senderWidget->value();
		if (scaleGroup[objectNumber] < sliderGroup[objectNumber -1]->value()) {
			sliderGroup[objectNumber - 1]->setValue(senderWidget->value() - 1);
		}
	}
	lcdGroup[objectNumber]->display(scaleGroup[objectNumber]);
	makeScaleList.clear();
	int scale = sliderGroup[0]->value();
	int scaleMax = sliderGroup[1]->value();
	while (scale<=scaleMax)
	{
		makeScaleList.append(scale);
		scale = scale * makeScaleRatio;
	}
}


void FastGMatchApp::startGMatchGPU() {

	QString fileName = QFileDialog::getOpenFileName(NULL, "dat¥Õ¥¡¥¤¥ë", ".", "dat(*.dat)");
	if (fileName == NULL) {
		return;
	}
	mtThread->fileName = fileName;
	mtThread->scale = scaleGroup[2];
	mtThread->scaleMax = scaleGroup[3];
	qmtThread[0]->start();
	qmtThread[0]->quit();
	qmtThread[0]->wait();
}


void FastGMatchApp::startGMakeGPU() {
	if (imageBuffer.cols == 0 || imageBuffer.rows == 0) {
		dealErrorMessage("Error","»­Ïñ¤ò¥í©`¥É¤·¤Æ¤¯¤À¤µ¤¤");
		return;
	}

	QString fileName = QFileDialog::getSaveFileName(NULL,"Save",".","dat(*.dat)");
	if (fileName == NULL) {
		return;
	}
	cv::Mat imgMat;
	cv::cvtColor(imageBuffer,imgMat , CV_BGR2RGB);
	mkThread->grayimg = &imgMat;
	mkThread->fileName = fileName;
	mkThread->scale = scaleGroup[0];
	mkThread->scaleMax = scaleGroup[1];
	qmtThread[1]->start();
	qmtThread[1]->quit();
	qmtThread[1]->wait();
}


void FastGMatchApp::loadImage() {
	QString fileName= QFileDialog::getOpenFileName(NULL, "»­Ïñ", ".","image(*.jpg)");
	if (fileName == NULL) {
		return;
	}
	qlistMat.clear();
	QByteArray str_arr = fileName.toLocal8Bit();
	const char* c_str = str_arr.constData();


	cv::Mat imgMat = cv::imread(c_str);
	cv::cvtColor(imgMat, imageBuffer, CV_RGB2BGR);
	QImage img;
	switch (imgMat.type())
	{
	case CV_8UC4:
	{
		img = QImage(imageBuffer.data,
			imageBuffer.cols, imageBuffer.rows,
			static_cast<int>(imageBuffer.step),
			QImage::Format_ARGB32);
	}
	case CV_8UC3:
	{
		img = QImage(imageBuffer.data,
			imageBuffer.cols, imageBuffer.rows,
			static_cast<int>(imageBuffer.step),
			QImage::Format_RGB888);
	}
	default:
		break;
	}

	GraphicsProcess gp;
	qlistMat =  gp.changeGraph(imgMat);
	//Mat a(qlistMat[8]);
	//img = QImage(a.data, a.cols, a.rows, static_cast<int>(a.step),QImage::Format_Grayscale8 );
	imageBufferPix = QPixmap::fromImage(img);
	ui->imageLabelMake->setPixmap(imageBufferPix.scaled(ui->imageLabelMake->size().width(), ui->imageLabelMake->size().height()));
}



void FastGMatchApp::resizeEvent(QResizeEvent *event) {
	ui->imageLabelMake->setPixmap(imageBufferPix.scaled(ui->imageLabelMake->size().width(), ui->imageLabelMake->size().height()));
}


void FastGMatchApp::dealErrorMessage(QString title, QString message) {
	QMessageBox::about(this, title, message);
}

void FastGMatchApp::loadChangedImage()
{
	if (!qlistMat.isEmpty()) {
		gcv->showImage(qlistMat,makeScaleList);
		gcv->show();
	}
}


