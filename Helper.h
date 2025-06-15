#pragma once

#include "base/Layer.h"
#include "base/Types.h"
#include "base/LearningRule.h"
#include "networks/Hopfield.h"
#include "networks/Perceptron.h"
#include <opencv2/opencv.hpp>

#include "base/Types.h"


Pattern png_to_bits(const std::string &filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat resizedImage;

    cv::resize(img, resizedImage, cv::Size(32, 32));

    Pattern pattern;
    pattern.reserve(static_cast<size_t>(resizedImage.rows)
                    * static_cast<size_t>(resizedImage.cols));

    for (size_t r = 0; r < static_cast<size_t>(resizedImage.rows); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(resizedImage.cols); ++c) {
            auto pixel = resizedImage.at<uchar>(static_cast<int>(r),
                                                static_cast<int>(c));
            pattern.emplace_back(
                //pixel > 128 ? 1.0 : -1.0); // Convert to bipolar representation
                pixel > 128 ? 1.0f : 0.0f); // Convert to binary representation
        }
    }
    return pattern;
}

void print(const Pattern &p)
{
    size_t idx = 0;
    for (auto i : p) {
        if (idx == 32) {
            std::cout << std::endl;
            idx = 0;
        }
        idx++;
        std::cout << i;
        //std::cout << ((i > 0) ? " " : "x");
    }
    std::cout << std::endl;
}

void hopfieldNetwork()
{
    Patterns patterns{};

    //patterns.emplace_back(png_to_bits("../Misc/bart.png"));
    //patterns.emplace_back(png_to_bits("../Misc/homer.png"));
    //patterns.emplace_back(png_to_bits("../Misc/marge.png"));
    //patterns.emplace_back(png_to_bits("../Misc/meg.png"));
    //patterns.emplace_back(png_to_bits("../Misc/grandpa.png"));
    //patterns.emplace_back(png_to_bits("../Misc/lisa.png"));
    //Pattern p(png_to_bits("../Misc/homer_defect.png"));

    // Example: create a 4-bit pattern and add it to patterns
    Pattern pattern4 = {1.0, -1.0, 1.0, -1.0};
    patterns.push_back(pattern4);

    // Use the same pattern as input for recall
    Pattern p = pattern4;

    auto N = patterns.at(0).size();

    Hopfield net(N);

    net.learn(patterns);

    auto ret = net.infer(p);

    print(p);
    std::cout << "Recall result: " << std::endl;
    print(ret);
    std::cout << "Hello, Hopfield Network!" << std::endl;
}

void perceptronNetwork()
{
    Patterns inputs
        = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

    Patterns labels = {
        {0.0}, // 0 AND 0
        {0.0}, // 0 AND 1
        {0.0}, // 1 AND 0
        {1.0}  // 1 AND 1
    };

    size_t in = inputs.at(0).size();
    size_t out = labels.at(0).size();
    Perceptron net(in, out);

    net.learn(inputs, labels);

    std::cout << "Perceptron Network trained!" << std::endl;
    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        print(input);
        print(output);
    }
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

