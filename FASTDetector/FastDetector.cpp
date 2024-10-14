#include "FastDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>


FastDetector::FastDetector(const cv::Mat& image, const int& threshold) {

	this->threshold = threshold;
	this->grayscaleImage = image;
	this->width = this->grayscaleImage.cols;
	this->height = this->grayscaleImage.rows;

}
const bool FastDetector::isDarker(const uchar& currPixel, const uchar& testPixel) const {

	return (testPixel - threshold > currPixel);
}
const bool FastDetector::isBrighter(const uchar& currPixel, const uchar& testPixel) const {

	return (testPixel + threshold < currPixel);
}


// OpenCV does this differently by using sum of absolute differences and compares the result to a threshold, I followed the original proposal
// Verifies the consistency of an arc in the Brescham circle by expanding from both sides until an undesirable value is detected, naive approach.
const bool FastDetector::isValidArc(const int& startIndex, const int& y, const int& x, const bool& isPbrighter) const {

	int iter = 1;
	int clockWise = startIndex, counterClockWise = startIndex;
	bool runClockwise = true;
	bool runCClockwise = true;
	const uchar* currPixel = grayscaleImage.ptr<uchar>(y, x);
	int currArcLen = 1; // one of 4 main pts is marked as start point of arc and thus we start with length 1
	
	while (runClockwise || runCClockwise) {

		if ((currArcLen + iter - circleLen) > arcLength) {
			return false;
		}
		//Right 
		if (runClockwise) {
			clockWise = (startIndex + iter) % circleLen;
			const uchar* testPixel = grayscaleImage.ptr<uchar>(y + circle[clockWise].y, x + circle[clockWise].x);

			if ((isBrighter(*currPixel, *testPixel) && isPbrighter) || (isDarker(*currPixel, *testPixel) && !isPbrighter)) {
				currArcLen++;

				if (currArcLen >= arcLength) {
					return true;
				}
			}
			else {
				runClockwise = false;
			}
		}

		//Left 
		if (runCClockwise) {
			counterClockWise = (startIndex - iter + circleLen) % circleLen;
			const uchar* testPixel = grayscaleImage.ptr<uchar>(y + circle[counterClockWise].y, x + circle[counterClockWise].x);

			if ((isBrighter(*currPixel, *testPixel) && isPbrighter) || (isDarker(*currPixel, *testPixel) && !isPbrighter)) {
				currArcLen++;

				if (currArcLen >= arcLength) {
					return true;
				}
			}
			else {

				runCClockwise  = false;
			}
		}
		iter++;
	}

	return false;
}
//rewrite without count? 
const bool FastDetector::isCorner(const int& y, const int& x) const{
	const uchar* currPixel = grayscaleImage.ptr<uchar>(y, x);

	bool isYmirrored = false;
	bool isRightCutset = false;
	bool isTopCutset = false;
	bool isPbrighter = false; // false means its dark true means its bright
	
	int darker;
	int brighter;

	// Establishment of the interest point 
	///////////////////////////////////
	const uchar* p0 = grayscaleImage.ptr<uchar>(y + circle[0].y) + (x + circle[0].x);
	const bool isP0Brighter = isBrighter(*currPixel, *p0);
	const bool isP0Darker = isDarker(*currPixel, *p0);
	if (isP0Brighter || isP0Darker) {
		isTopCutset = true;
	}
	///////////////////////////////////
	const uchar* p8 = grayscaleImage.ptr<uchar>(y + circle[8].y) + (x + circle[8].x);
	const bool isP8Brighter = isBrighter(*currPixel, *p8);
	const bool isP8Darker = isDarker(*currPixel, *p8);

	darker = isP0Darker + isP8Darker; 
	brighter = isP0Brighter + isP8Brighter;
	if (brighter <= 0 && darker <= 0) {
		return false;
	}
	else if (brighter == 2 || darker == 2) {
		isYmirrored = true; // 
	} //try dpo
	/////////////////////////////////
	const uchar* p4 = grayscaleImage.ptr<uchar>(y + circle[4].y) + (x + circle[4].x);
	const bool isP4Brighter = isBrighter(*currPixel, *p4);
	const bool isP4Darker = isDarker(*currPixel, *p4);

	darker += isP4Darker;
	brighter += isP4Brighter;

	if (brighter == 3 || darker == 3) {
		//isRightCutset = true;
		if (isYmirrored) {
			return isValidArc(4, y, x, brighter == 3);
		}

	}

	const uchar* p12 = grayscaleImage.ptr<uchar>(y + circle[12].y) + (x + circle[12].x);
	const bool isP12Brighter = isBrighter(*currPixel, *p12);
	const bool isP12Darker = isDarker(*currPixel, *p12);
	darker += isP12Darker;
	brighter += isP12Brighter;

	if (!(brighter >= 3 || darker >= 3)) { // No need to look further, impossible
		return false; 
	}

	// Verification of interest point
	if (brighter >= 3) {
		isPbrighter = true;
	}

	if (isYmirrored) {
		//if (isRightCutset) {
			//p0, p4, p8
			//return isValidArc(4, y, x, isPbrighter); // start from p0? 
		//}
		//p0, p12, p8
			return (isValidArc(12, y, x, isPbrighter)); //start from p8? 
		}
	else {
		if (isTopCutset) {
			// p12, p0, p4
			return (isValidArc(0, y, x, isPbrighter)); //start from p12
		}    
		//p12, p8, p4;
		return (isValidArc(8, y, x, isPbrighter));	// start from p4
	}
	
}
void FastDetector::extractFeatures(std::vector<cv::KeyPoint>& keypoints) const {

	// Preallocate a local keypoints vector for each thread
	std::vector<std::vector<cv::KeyPoint>> localKeypoints(omp_get_max_threads());

	#pragma omp parallel for
	for (int y = radius; y < height - radius; y++) {
		for (int x = radius; x < width - radius; x++) {
			if (isCorner(y, x)) {

				// Sobel Kernel on interest point
				const double Ix = ((-1 * (*grayscaleImage.ptr<uchar>(y + 1, x - 1))) + (1 * (*grayscaleImage.ptr<uchar>(y + 1, x + 1)))
					+ (-1 * (*grayscaleImage.ptr<uchar>(y - 1, x - 1))) + (1 * (*grayscaleImage.ptr<uchar>(y - 1, x + 1)))
					+ (-2 * (*grayscaleImage.ptr<uchar>(y, x - 1))) + (2 * (*grayscaleImage.ptr<uchar>(y, x + 1)))) / 255.0;

				const double Iy = ((-1 * (*grayscaleImage.ptr<uchar>(y - 1, x - 1))) + (-2 * (*grayscaleImage.ptr<uchar>(y - 1, x))) + (-1 * (*grayscaleImage.ptr<uchar>(y - 1, x + 1)))
					+ (1 * (*grayscaleImage.ptr<uchar>(y + 1, x - 1))) + (2 * (*grayscaleImage.ptr<uchar>(y + 1, x))) + (1 * (*grayscaleImage.ptr<uchar>(y + 1, x + 1)))) / 255.0;

				// Harris matrix 
				const double Ixx = Ix * Ix;
				const double Iyy = Iy * Iy;
				const double Ixy = Ix * Iy;

				// Harris cornerness response
				const double detM = (Ixx * Iyy) - (Ixy * Ixy);
				const double traceM = Ixx + Iyy;
				const double harrisResponse = std::abs(detM - 0.04f * (traceM * traceM));

				if (harrisResponse > 0) {
					cv::KeyPoint keypoint;
					keypoint.pt = cv::Point2f(x, y);
					keypoint.size = 6;  // Circle size

					// Get the thread ID
					int threadId = omp_get_thread_num();
					localKeypoints[threadId].push_back(keypoint); // Add to local vector
				}
			}
		}
	}

	// Critical section: Merge all local vectors into the shared keypoints vector
	for (const auto& localVec : localKeypoints) {
		keypoints.insert(keypoints.end(), localVec.begin(), localVec.end());
	}
}