#include <opencv2/opencv.hpp>
#include <iostream>
#include "FastDetector.h"
#include <chrono>  
#include "BRIEF.h"
#include <bitset>



enum class DetectorType {
    Custom,
    OpenCV
};

static void processImage(const std::string& imagePath, const int& fastThreshold, DetectorType detectorType) {

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Image " << imagePath << " could not be loaded!" << std::endl;
        return;
    }

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    grayImage.convertTo(grayImage, CV_8UC1); // For custom detector
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat outputImage;

    BRIEF<256> descriptor(6);
    auto result = ov::generatePattern<256>();

    keypoints.reserve(grayImage.rows * grayImage.cols);
    try {
        auto start = std::chrono::high_resolution_clock::now();

        if (detectorType == DetectorType::Custom) {
            FastDetector customDetector(grayImage, fastThreshold);
         
            customDetector.extractFeatures(keypoints);
            std::vector<std::bitset<256>> descriptors = descriptor.generateDescriptor(keypoints, grayImage);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedCustom = end - start;
            std::cout << "Custom FastDetector feature extraction for " << imagePath << " took " << elapsedCustom.count() << " seconds." << std::endl;
        }
        else if (detectorType == DetectorType::OpenCV) {
            // Use OpenCV FAST detector
            cv::FAST(grayImage, keypoints, fastThreshold, false); // true for non-maxima suppression
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "OpenCV FAST feature extraction for " << imagePath << " took " << elapsed.count() << " seconds." << std::endl;
        }

        // Draw keypoints on the imaged
        cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); // Green color
        cv::imshow(detectorType == DetectorType::Custom ? "Custom Detected Features: " : "Detected Features: " + imagePath, outputImage);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: Feature extraction failed: " << e.what() << std::endl;
    }
}

int main() {
    int fastThreshold = 70;  // You can adjust this threshold as needed


    processImage("S1-im2.png", fastThreshold, DetectorType::Custom);
    processImage("S1-im2.png", fastThreshold, DetectorType::OpenCV);

    cv::waitKey(0);


    return 0;
}