#pragma once

#include <QMainWindow>
#include "ui_GraphcisChangeView.h"
#include <qlist.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <qgraphicsscene.h>
#include <qgraphicsitem.h>


class GraphcisChangeView : public QMainWindow
{
	Q_OBJECT

public:
	GraphcisChangeView( QWidget *parent = Q_NULLPTR);
	void showImage(QList<cv::Mat> qlistMat,QList<int> scaleList);
	~GraphcisChangeView();

private:
	Ui::GraphcisChangeView *ui = new Ui::GraphcisChangeView();
	QGraphicsScene *scene;
	QList<QGraphicsPixmapItem*> qlistPixitem;
};
