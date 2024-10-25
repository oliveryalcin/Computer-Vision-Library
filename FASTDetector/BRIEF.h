#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <bitset>
#include <iostream>


template <int N>
class BRIEF {
private:
	using PointPair = std::pair<std::pair<int, int>, std::pair<int, int>>;
	std::vector<PointPair> comparisonPattern;


public:
	BRIEF(const int& patchSize);
	const std::vector <std::bitset<N>> generateDescriptor(std::vector<cv::KeyPoint> keypoints, const cv::Mat& image) const;
};

// Constructor will take patch type (circle, triangle, square etc..) and also patch size to generate distribution correctly
/*
	Generates patch pattern using uniform distribution.
	Should be called once to initialize patch size, type and patch pattern used for descriptor generation
*/
template<int N>
BRIEF<N>::BRIEF(const int& patchSize) {
	static std::random_device rd;
	static std::mt19937 gen(rd());

	// Distribution setup with typename for type-dependent context
	typename std::uniform_int_distribution<> distr(-patchSize/2, patchSize/2);

	for (int i = 0; i < N; i++) {
		PointPair pairOfPts(
			std::make_pair(distr(gen), distr(gen)),
			std::make_pair(distr(gen), distr(gen))
		);
		comparisonPattern.push_back(pairOfPts);
	}
}


/*
		Returns Binary descriptor array used for keypoint matching
	// https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
	// https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126544
*/
template <int N>
const std::vector <std::bitset<N>> BRIEF<N>::generateDescriptor(std::vector<cv::KeyPoint> keypoints, const cv::Mat& image) const{ // Generates pairs of points to compare for a BRIEF descriptor uniformally.

	std::bitset<N> bits;
	std::vector <std::bitset<N>> keypointDescriptors;

	// this is all nonsense, just randomly sample for nd times (128,256,512)
	int kpointSize = keypoints.size();
	for (int i = 0; i < kpointSize; i++) {
		const int x = static_cast<int>(keypoints.at(i).pt.x);
		const int y = static_cast<int>(keypoints.at(i).pt.y);

		for (int j = 0; j < N; j++) {
			PointPair pairOfPts;
			std::pair<int, int> point1;
			std::pair<int, int> point2;

			point1 = comparisonPattern.at(j).first;
			point2 = comparisonPattern.at(j).second;

			const uchar intensityA = *image.ptr<uchar>(y + point1.second, x + point1.first);
			const uchar intensityB = *image.ptr<uchar>(y + point2.second, x + point2.first);

			bits.set(j, intensityA > intensityB);  // Set bit at position i based on intensity comparison
		}
		keypointDescriptors.push_back(bits); // Generate BRIEF descriptor given FAST feature points
	}
	return keypointDescriptors;

}
