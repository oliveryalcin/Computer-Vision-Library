#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
//#include <vector>

class FastDetector
{
private:
	cv::Mat colorImage;
	cv::Mat grayscaleImage;
	cv::Mat detectedFeatures;
	int width, height, arcLength = 12, threshold = 10;
	const int radius = 3;
	bool isNonMax = false;



	/*
	typedef struct{
		int darkCount = 0;
		int brightCount = 0;
		int similarCount = 0; 
	}CountData;
	*/


	typedef struct   {
		int x, y;
	}coordinate;
	static const int circleLen = 16;
	const coordinate circle[circleLen] = {
		{0,3}, {1,3}, {2,2}, {3,1},
		{3,0}, {3,-1}, {2,-2}, {1,-3}, //reflection starts here, can halve array by just multiplying (-)
		{0,-3}, {-1,-3}, {-2,-2}, {-3,-1},
		{-3,0}, {-3,1}, {-2,2}, {-1,3},
	};
	void markArcAsProcessed(int startIdx, int endIdx, std::vector<bool>& processed, std::vector<int>& arcArray);

	bool isDarker(const uchar& currPixel, const uchar& testPixel, const int& threshold) const;
	bool isBrighter(const uchar& currPixel, const uchar& testPixel, const int& threshold) const;
	bool isCorner(const int& x, const int& y);
	void fastArcSearch();
	void FastR();


public:
	
	FastDetector();
	FastDetector(const bool& isNonMax, const int& circleRadius, const int& arcLength);
	

	cv::Mat extractFeatures();


};

