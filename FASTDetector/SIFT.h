#pragma once
#include <opencv2/opencv.hpp>

class SIFT {
private:
	cv::Mat imDx;
	cv::Mat imDy;
	cv::Mat image;
	//cv::Mat 
public:

	SIFT();
	SIFT(cv::Mat image);



};

