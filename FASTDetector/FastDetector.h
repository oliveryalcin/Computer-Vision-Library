#ifndef FASTCustom
#define FASTCustom

#include <opencv2/opencv.hpp>
#include <iostream>
#include <bitset>
#include <vector>
#include <utility> // For std::pair
#include <random>  // For random number generation
#include <iostream> // For std::cout, std::endl


typedef std::pair<std::pair<int, int>, std::pair<int, int>> PointPair; //pairing of points



typedef struct {
	int x, y;
}coordinate;

//template <int T>
class FastDetector {
private:
	cv::Mat grayscaleImage;
	int width, height;
	int threshold;
	const int arcLength = 9;
	const int radius = 3;
	std::vector <std::bitset<256>> keypointDescriptors;
	std::vector<PointPair> comparisonPattern;
	int descriptorSize;


	// Write Function to initialize
	static const int circleLen = 16;
	const coordinate circle[circleLen] = {
		{0, 3}, {1, 3}, {2, 2}, {3, 1},
		{3, 0}, {3, -1}, {2, -2}, {1, -3}, // First octant
		{0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, // Second octant
		{-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}, // Third and fourth octants
	};
	//static std::vector<PointPair> comparisonPattern;

	const bool isDarker(const uchar& currPixel, const uchar& testPixel) const;
	const bool isBrighter(const uchar& currPixel, const uchar& testPixel) const;
	const bool isCorner(const int& y, const int& x) const;
	const bool isValidArc(const int& startIndex, const int& y, const int& x, const bool& isPbrighter) const;
	//void calculateGradients(const int& y, const int& x, double& Ix, double& Iy) const;
	const double computeHarrisResponse(const double& Ix, const double& Iy) const;
	void generateDescriptor(const int& nd, const int& x, const int& y, const double& angle);
	double computeCentroidOrientation(const int& x, const int& y);
	const double gradX(const int& x, const int& y) const;
	const double gradY(const int& x, const int& y) const;
	void savePixelCoordinates(const std::vector<std::pair<int, int>>& brighterPixels, const std::vector<std::pair<int, int>>& darkerPixels) const;

public:



	/*
		Generates uniformally distributed list of point-pairs
		Point pairs used to encode comparisons consistently throughout the image
		Feed pattern to Descriptor	
	*/

	
	FastDetector(const cv::Mat& image, const int& threshold, std::vector<PointPair> pattern, const int& descriptorSize);
	void extractFeatures(std::vector<cv::KeyPoint>& keypoints);


};
namespace ov { // Namespace for all computer vision related functions
	// Template function to generate a pattern
	template <int N>
	const std::pair<std::vector<PointPair>, int> generatePattern() {
		std::vector<PointPair> pattern;

		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_int_distribution<int> distr(-15, 15);

		for (int i = 0; i < N; i++) {
			PointPair pairOfPts(
				std::make_pair(distr(gen), distr(gen)), 
				std::make_pair(distr(gen), distr(gen))  
			);
			pattern.push_back(pairOfPts); 
		}
		return std::make_pair(pattern, N);
	}

} // namespace ov
#endif
