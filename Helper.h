#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include "Layer.h"

Pattern png_to_bits(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE); 
    cv::Mat resizedImage;

    cv::resize(img, resizedImage, cv::Size(32, 32));

    Pattern pattern;
    pattern.reserve(static_cast<size_t>(resizedImage.rows) * static_cast<size_t>(resizedImage.cols));
    
    for (size_t r = 0; r < static_cast<size_t>(resizedImage.rows); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(resizedImage.cols); ++c) {
            auto pixel = resizedImage.at<uchar>( static_cast<int>(r), static_cast<int>(c)); 
            pattern.emplace_back( pixel > 128 ? 1.0 : -1.0 ); // Convert to bipolar representation
        }
    }
    return pattern;
}


void print(const Pattern &p)
{

    size_t idx = 0;
    for (auto i :  p) {
        if(idx == 32) {
            std::cout << std::endl;
            idx = 0;
        }
        idx++;
        std::cout << i;
        //std::cout << ((i > 0) ? " " : "x");
    }
    std::cout << std::endl;
}

/*

// Compute Hamming distance between two patterns
int hamming_distance(const Pattern& a, const Pattern& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++dist;
    }
    return dist;
}

    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            std::cout << "Hamming(" << i << "," << j << ") = "
                      << hamming_distance(patterns[i], patterns[j]) << std::endl;
        }
    }
*/

#endif // HELPER_H