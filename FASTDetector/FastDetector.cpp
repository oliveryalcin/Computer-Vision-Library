#include "FastDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

FastDetector::FastDetector() {

    cv::Mat img = cv::imread("Lenna.png");
	this->colorImage = img;
    cv::cvtColor(this->colorImage, this->grayscaleImage, cv::COLOR_BGR2GRAY);
    this->width = this->grayscaleImage.cols;
	this->height = this->grayscaleImage.rows;
	this->detectedFeatures = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));

}
bool FastDetector::isDarker(const uchar& currPixel, const uchar& testPixel, const int& threshold) const{

    return (currPixel - testPixel >= threshold);
}
bool FastDetector::isBrighter(const uchar& currPixel, const uchar& testPixel, const int& threshold) const{

    return (testPixel - currPixel >= threshold);
}
// Sequential Breadth First Consistency Jam
void FastDetector::fastArcSearch() {

}
//High Speed Test
void FastDetector::FastR() {

}
// use a doubly linked queue
bool FastDetector::isCorner(const int& y, const int& x){
    
    for (int startIdx = 0; startIdx < circleLen; startIdx++) { // iterate circle for initial starting points
        int darkCount = 0, brightCount = 0, similarCount = 0;
        int remainingPixels = 0;
        for (int i = 0; i < circleLen; i++) {
            int idx = (startIdx + i) % circleLen;  // Wrap around
            int dx = circle[idx].x;
            int dy = circle[idx].y;
            uchar currPixel = grayscaleImage.at<uchar>(y, x);  // Current center pixel
            uchar testPixel = grayscaleImage.at<uchar>(y + dy, x + dx);  // Pixel on the circle

            // Check if the current pixel is darker
            if (isDarker(currPixel, testPixel, threshold)) {
                darkCount++;
                brightCount = 0;  // Reset bright count


                if (darkCount >= arcLength) {
                    return true;
                }
            }
            else if (isBrighter(currPixel, testPixel, threshold)) {
                darkCount = 0;  
                brightCount++;

                if (brightCount >= arcLength) {
                    return true;
                }
            }
            else {
                darkCount = 0;
                brightCount = 0;
                similarCount++; // useless pixels so excluding them yields performance gains
            }
            // Must be revised, overcounting or undercounting
            remainingPixels = 16 - (i + 1) - similarCount;
            if ((remainingPixels + darkCount < arcLength) && (remainingPixels + brightCount  < arcLength)) {
                break;
            }
        }
        if ((remainingPixels + darkCount < arcLength) && (remainingPixels + brightCount < arcLength)) {
            break;
        }
    }

    return false;
}

cv::Mat FastDetector::extractFeatures() {
	
	for (int y = radius; y < height - radius; y++) {
		for (int x = radius; x < width - radius; x++) {
			uchar& pixelValue = detectedFeatures.at<uchar>(y, x);
			if (isCorner(y,x)) {
				pixelValue = 255;
			}
		}
	}
	return detectedFeatures;
}