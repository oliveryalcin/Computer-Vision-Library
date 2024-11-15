#include "FastDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <string.h>
#include <bitset>
#include <algorithm>
#include <fstream>
#include <vector>
#include <utility> // For std::pair
// Butun ptrlari duzelt aminakoyayim


FastDetector::FastDetector(const cv::Mat& image, const int& threshold){

	this->threshold = threshold;
	this->grayscaleImage = image;
	this->width = this->grayscaleImage.cols;
	this->height = this->grayscaleImage.rows;

	this->descriptorSize = descriptorSize;


}

const bool FastDetector::isCorner(const int& y, const int& x) const {
	const uchar* currPixel = grayscaleImage.ptr<uchar>(y, x);

	bool isYmirrored = false, isRightCutset = false, isTopCutset = false, isPbrighter = false;
	int darker = 0, brighter = 0;

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
		isRightCutset = true;
		if (isYmirrored) {
			//return isValidArc(4, y, x, brighter == 3);

		}

	}

	const uchar* p12 = grayscaleImage.ptr<uchar>(y + circle[12].y) + (x + circle[12].x);
	const bool isP12Brighter = isBrighter(*currPixel, *p12);
	const bool isP12Darker = isDarker(*currPixel, *p12);
	darker += isP12Darker;
	brighter += isP12Brighter;


	// Verification of interest point
	if (brighter >= 3) {
		isPbrighter = true;
	}
	if (brighter >= 3 || darker >= 3) {
		return true;
	}
	return false;

	if (isYmirrored) {
		if (isRightCutset) {
			//p0, p4, p8
			return isValidArc(4, y, x, isPbrighter); // start from p0? 
		}
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
void FastDetector::extractFeatures(std::vector<cv::KeyPoint>& keypoints) {

	//#pragma omp parallel for
	for (int y = radius; y < height - radius; y++) {
		for (int x = radius; x < width - radius; x++) {
			if (isCorner(y, x)) {

				const double dX = gradX(x, y);
				const double dY = gradY(x, y);
			
				const double harrisResponse = computeHarrisResponse(dX, dY);

				if (harrisResponse >= 100000) {
					cv::KeyPoint keypoint;
					keypoint.pt = cv::Point2f(x, y);
					keypoint.size = 6;  
					keypoints.push_back(keypoint);

				}
			}
		}
	}

	//return keypointDescriptors;
}
//https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
// https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126544
/*
void FastDetector::generateDescriptor(const int& nd, const int& x, const int& y, const double& angle) { // Generates pairs of points to compare for a BRIEF descriptor uniformally.

	std::bitset<256> bits;

	for (int i = 0; i < nd; i++) {
		PointPair pairOfPts;
		std::pair<int, int> point1;
		std::pair<int, int> point2;

		point1 = comparisonPattern.at(i).first;
		point2 = comparisonPattern.at(i).second;


		const uchar intensityA = *grayscaleImage.ptr<uchar>(y + point1.second, x + point1.first);
		const uchar intensityB = *grayscaleImage.ptr<uchar>(y + point2.second, x + point2.first);

		bits.set(i, intensityA > intensityB);  // Set bit at position i based on intensity comparison
	}
	keypointDescriptors.push_back(bits); // Generate BRIEF descriptor given FAST feature points
	// Ordering of keypoints is parallel to the generated keypoints so the order of descriptor patches is related to the patch created by the keypoint
}
*/

const double FastDetector::gradX(const int& x, const int& y) const { // Gradient in X direction
	// Sobel Kernel on interest point
	const double gradient =
		((-1.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y + 1, x - 1))) +
			(1.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y + 1, x + 1))) +
			(-1.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y - 1, x - 1))) +
			(1.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y - 1, x + 1))) +
			(-2.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y, x - 1))) +
			(2.0 / 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y, x + 1))));


	return gradient; // Return the computed gradient if within range
}

const double FastDetector::gradY(const int& x, const int& y) const { // Gradient in Y direction
	// Sobel Kernel on interest point
	const double gradient =
		((-1.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y - 1, x - 1))) +
		(-2.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y - 1, x))) +
		(-1.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y - 1, x + 1))) +
		(1.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y + 1, x - 1))) +
		(2.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y + 1, x))) +
		(1.0/ 8 * static_cast<double>(*grayscaleImage.ptr<uchar>(y + 1, x + 1))));


	return gradient; //normalized gradient instead normalize harris cornerness response l8r
}
const double FastDetector::computeHarrisResponse(const double& Ix, const double& Iy) const {

	// Harris matrix 
	const double Ixx = Ix * Ix;
	const double Iyy = Iy * Iy;
	const double Ixy = Ix * Iy;

	// Harris cornerness response
	const double detM = (Ixx * Iyy) - (Ixy * Ixy);
	const double traceM = Ixx + Iyy;
	//const double val = std::abs(detM - 0.04f * (traceM * traceM));
	//normalize meausre
	
	return std::abs(detM - 0.04f * (traceM * traceM));
}

/*

static const int hammingDistance(const std::bitset<128>& desc1, const std::bitset<128>& desc2) { //Score function for given descriptor
	return (desc1 ^ desc2).count(); // XOR the two descriptors and count the differing bits
}
*/
const bool FastDetector::isDarker(const uchar& currPixel, const uchar& testPixel) const {

	int adjustedCurrPixel = std::min(int(currPixel) + threshold, 255);
	return (testPixel > adjustedCurrPixel);
}

const bool FastDetector::isBrighter(const uchar& currPixel, const uchar& testPixel) const {

	// Promote currPixel to int to avoid underflow before subtracting threshold
	int adjustedCurrPixel = std::max(static_cast<int>(currPixel) - threshold, 0);
	return (int(testPixel) < adjustedCurrPixel);
}
 
// OpenCV does this differently by using sum of absolute differences and compares the result to a threshold, I followed the original proposal
// Verifies the consistency of an arc in the Brescham circle by expanding from both sides until an undesirable value is detected, naive approach.
const bool FastDetector::isValidArc(const int& startIndex, const int& y, const int& x, const bool& isPbrighter) const {

	int leftIndex = startIndex; // decrement
	int rightIndex = startIndex; // increment
	int upperIndex = startIndex;
	const uchar* currPixel = grayscaleImage.ptr<uchar>(y, x);
	int count = 0;
	int rightCount = 0;
	int leftCount = 0;

	bool stopDecrement = false;
	bool stopIncrement = false;


	for(int i = 0; i < 16; i++){ // general counter i < 9 upper bound because of arc but should quit for diff values if logic correct check with debugger

		if (i > 4) {
			
		}

		if (leftIndex - 1 < 0 || stopDecrement) { // should i start increment:
			leftIndex++;
			stopDecrement = true;
		}
		else {
			leftIndex--;
		}
		if (rightIndex + 1 == 16 || stopIncrement) {
			stopIncrement = true;
			rightIndex--;
		}
		else {
			rightIndex++;
		}

		

		const uchar* leftPixel = grayscaleImage.ptr<uchar>(y + circle[leftIndex].y, x + circle[leftIndex].x);
		const uchar* righttPixel = grayscaleImage.ptr<uchar>(y + circle[rightIndex].y, x + circle[rightIndex].x);


		if (!(isBrighter(*currPixel, *leftPixel) && isPbrighter) || (isDarker(*currPixel, *leftPixel) && !isPbrighter) && i > 4) {
			return false;
		}
		else{
			//count++;
			leftCount++;

		}
		if (!(isBrighter(*currPixel, *righttPixel) && isPbrighter) || (isDarker(*currPixel, *righttPixel) && !isPbrighter) && i > 4) {
			return false;
		}
		else {
			//count++;
			rightCount++;
		}
		

	

	}

	// Redo this shit
	/* 
	int iter = 1;
	int clockWise = startIndex, counterClockWise = startIndex;
	bool runClockwise = true;
	bool runCClockwise = true;
	const uchar* currPixel = grayscaleImage.ptr<uchar>(y, x);
	int currArcLen = 1; // Start with length 1 (main point)

	// Store the coordinates of the pixels
	std::vector<std::pair<int, int>> brighterPixels;
	std::vector<std::pair<int, int>> darkerPixels;

	while (runClockwise || runCClockwise) {

		// Right (Clockwise)
		if (runClockwise) {
			clockWise = (startIndex + iter) % (circleLen);

			const uchar* testPixel = grayscaleImage.ptr<uchar>(y + circle[clockWise].y, x + circle[clockWise].x);

			if ((isBrighter(*currPixel, *testPixel) && isPbrighter) || (isDarker(*currPixel, *testPixel) && !isPbrighter)) {

				if (isBrighter(*currPixel, *testPixel)) {
					// Save brighter pixel coordinates
					brighterPixels.emplace_back(x + circle[clockWise].x, y + circle[clockWise].y);
				}
				else {
					// Save darker pixel coordinates
					darkerPixels.emplace_back(x + circle[clockWise].x, y + circle[clockWise].y);
				}

				if (++currArcLen >= arcLength) {
					return true;
				}
			}
			else {
				runClockwise = false;
			}
		}

		// Left (Counterclockwise)
		if (runCClockwise) {
			counterClockWise = (startIndex + 15 + iter) % (circleLen);

			const uchar* testPixel = grayscaleImage.ptr<uchar>(y + circle[counterClockWise].y, x + circle[counterClockWise].x);

			if ((isBrighter(*currPixel, *testPixel) && isPbrighter) || (isDarker(*currPixel, *testPixel) && !isPbrighter)) {
				if (++currArcLen >= arcLength) {
					return true;
				}
			}
			else {
				runCClockwise = false;
			}
		}

		iter++;
	}
	*/
	return false;
}