#include "GraphcisChangeView.h"


GraphcisChangeView::GraphcisChangeView( QWidget * parent) : QMainWindow(parent)
{
	ui->setupUi(this);

	scene = new QGraphicsScene(this);

	ui->graphicsView->setScene(scene);


}

void GraphcisChangeView::showImage(QList<cv::Mat> qlistMat, QList<int> scaleList)
{
	int index = 0;
	qlistPixitem.clear();

	for (QList<cv::Mat>::iterator i = qlistMat.begin(); i != qlistMat.end(); ++i) {
		int row = index / 4;
		int col = index % 4;
		cv::Mat cImage(*i);
		cv::cvtColor(cImage, cImage, CV_GRAY2RGB);
		for (QList<int>::iterator j = scaleList.begin(); j != scaleList.end(); ++j) {
			cv::circle(cImage, cvPoint(cImage.cols / 2, cImage.rows / 2), *j, CV_RGB(255, 0, 0), 10);
		}
		QImage qimage(cImage.data, cImage.cols, cImage.rows, static_cast<int>(cImage.step), QImage::Format_RGB888);
		QPixmap pixImage = QPixmap::fromImage(qimage);
		QGraphicsPixmapItem *qgpi = scene->addPixmap(pixImage.scaled(200, 200));
		qgpi->setOffset(300 * col, 300 * row);
		qlistPixitem.append(qgpi);
		index++;
	}
}

GraphcisChangeView::~GraphcisChangeView()
{
}
