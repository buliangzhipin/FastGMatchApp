#pragma once

#include <QObject>

class MatchThread : public QObject
{
	Q_OBJECT

public:
	MatchThread(QObject *parent=nullptr);
	~MatchThread();
	QString fileName;
	int scale;
	int scaleMax;

public slots:
	void matchImage();

signals:
	void errorMessage(QString title,QString message);

};
