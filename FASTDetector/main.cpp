#include <opencv2/opencv.hpp>

#include <iostream>
#include "FastDetector.h"

#include <chrono>  // For high-resolution timing
#include <opencv2/opencv.hpp>  // Assuming you are using OpenCV

int main() {
    // Create an instance of FastDetector
    FastDetector detector;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Extract features
    cv::Mat img = detector.extractFeatures();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Display the detected features
    cv::imshow("Detected Features", img);

    // Wait for a key press to close the window
    cv::waitKey(0);

    // Calculate duration
    std::chrono::duration<double> elapsed = end - start;

    // Print out the time taken to extract features
    std::cout << "Feature extraction took " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
