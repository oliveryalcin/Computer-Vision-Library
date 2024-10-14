#ifndef FASTCustom
#define FASTCustom

#include <opencv2/opencv.hpp>
#include <iostream>

typedef struct {
	int x, y;
}coordinate;

class FastDetector
{
private:
	cv::Mat grayscaleImage;
	int width, height;
	int threshold;
	const int arcLength = 9;
	const int radius = 3;



	static const int circleLen = 16;
	const coordinate circle[circleLen] = {
		{0, 3}, {1, 3}, {2, 2}, {3, 1},
		{3, 0}, {3, -1}, {2, -2}, {1, -3}, // First octant
		{0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, // Second octant
		{-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, // Third and fourth octants
	};
	const bool isDarker(const uchar& currPixel, const uchar& testPixel) const;
	const bool isBrighter(const uchar& currPixel, const uchar& testPixel) const;
	const bool isCorner(const int& y, const int& x) const;

	const bool isValidArc(const int& startIndex, const int& y, const int& x, const bool& isPbrighter) const;
	void calculateGradients(const int& y, const int& x, double& Ix, double& Iy) const;
	const double computeHarrisResponse(const double& Ix, const double& Iy) const;

public:
	
	FastDetector(const cv::Mat& image, const int& threshold);
	void extractFeatures(std::vector<cv::KeyPoint>& keypoints) const;


};
#endif
