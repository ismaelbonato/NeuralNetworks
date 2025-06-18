#pragma once

#include "base/Layer.h"
#include "base/LearningRule.h"
#include "base/Types.h"
#include "networks/FeedForward.h"
#include "networks/Hopfield.h"
#include "networks/Perceptron.h"
#include <opencv2/opencv.hpp>

#include "base/Types.h"

Pattern png_to_bits(const std::string &filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat resizedImage;

    cv::resize(img, resizedImage, cv::Size(16, 16));

    Pattern pattern;
    pattern.reserve(static_cast<size_t>(resizedImage.rows)
                    * static_cast<size_t>(resizedImage.cols));

    for (size_t r = 0; r < static_cast<size_t>(resizedImage.rows); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(resizedImage.cols); ++c) {
            auto pixel = resizedImage.at<uchar>(static_cast<int>(r),
                                                static_cast<int>(c));
            pattern.emplace_back(
                //pixel > 128 ? 1.0 : -1.0); // Convert to bipolar representation
                pixel > 128 ? Scalar{1.0f} : Scalar{}); // Convert to binary representation
        }
    }
    return pattern;
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

    HopfieldLayer layer(std::make_shared<HebbianRule<Scalar>>(),
                        std::make_shared<StepPolarActivation<Scalar>>(),
                        N,
                        N);

    Hopfield net(layer.clone());

    net.learn(patterns);

    auto ret = net.infer(p);

    std::cout << p << std::endl;
    std::cout << "Recall result: " << std::endl;
    std::cout << ret << std::endl;
    std::cout << "Hello, Hopfield Network!" << std::endl;
}

void perceptronNetwork()
{
    Patterns inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

    Patterns labels = {
        {0.0}, // 0 AND 0
        {0.0}, // 0 AND 1
        {0.0}, // 1 AND 0
        {1.0}  // 1 AND 1
    };

    size_t in = inputs.at(0).size();
    size_t out = labels.at(0).size();

    DenseLayer layer(std::make_shared<PerceptronRule<Scalar>>(),
                     std::make_shared<SigmoidActivation<Scalar>>(),
                     in,
                     out);

    Perceptron net(layer.clone());

    net.learn(inputs, labels);

    std::cout << "Perceptron Network trained!" << std::endl;
    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
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

void feedForwardNetwork()
{
    /*    Patterns inputs;

    inputs.emplace_back(png_to_bits("../Misc/bart.png"));
    inputs.emplace_back(png_to_bits("../Misc/homer.png"));
    inputs.emplace_back(png_to_bits("../Misc/marge.png"));
    inputs.emplace_back(png_to_bits("../Misc/meg.png"));
    inputs.emplace_back(png_to_bits("../Misc/grandpa.png"));
    inputs.emplace_back(png_to_bits("../Misc/lisa.png"));

    Pattern p(png_to_bits("../Misc/meg.png"));

    Patterns labels = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, //bart
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, //homer
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, //marge
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, //meg
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, //grandpa
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};  //lisa
*/

    std::cout << "Manual Feedforward Network" << std::endl;

    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();

    DenseLayer layer1(rule, activation, 2, 4);
    DenseLayer layer2(rule, activation, 4, 2);
    DenseLayer layer3(rule, activation, 2, 1);

    //FeedforwardNetwork net(std::move(layers));
    Feedforward net(layer1);

    net.addLayers(layer2);

    net.addLayer(layer3);

    Patterns inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    Patterns labels = {{0.0}, {1.0}, {1.0}, {0.0}};

    net.learn(inputs, labels);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
        ;
    }
}