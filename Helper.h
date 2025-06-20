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
                pixel > 128 ? Scalar{1.0f}
                            : Scalar{}); // Convert to binary representation
        }
    }
    return pattern;
}

void hopfieldNetwork()
{
    Patterns patterns{};

    Pattern pattern = {1.0, -1.0, 1.0, -1.0};
    patterns.push_back(pattern);

    // Use the same pattern as input for recall
    LayerConfig config{
        .learningRule = std::make_shared<HebbianRule<Scalar>>(),
        .activation = std::make_shared<StepPolarActivation<Scalar>>(),
        .inputSize = patterns.at(0).size(),
        .outputSize = patterns.at(0).size(),
        .name = "Input",
        .type = "HopfieldLayer",
        .info = "info",
        .useBias = false
    };

    auto l = std::make_shared<HopfieldLayer>(config);

    Hopfield net(l);

    net.learn(patterns);

    auto ret = net.infer(pattern);

    std::cout << pattern << std::endl;
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

    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize =inputs.at(0).size(),
        .outputSize = labels.at(0).size(),
        .name = "Perceptron",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    auto l = std::make_shared<DenseLayer>(config);
    
    Perceptron net(l);

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

    LayerConfig config1{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 4,
        .name = "Input",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    LayerConfig config2{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 4,
        .outputSize = 2,
        .name = "Hidden Layer",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    LayerConfig config3{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 1,
        .name = "Output",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };


    auto l1 = std::make_shared<DenseLayer>(config1);
    auto l2 = std::make_shared<DenseLayer>(config2);
    auto l3 = std::make_shared<DenseLayer>(config3);

    //FeedforwardNetwork net(std::move(layers));
    Feedforward net({l1});

    net.addLayers({l2});

    net.addLayer(l3);

    Patterns inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    Patterns labels = {{0.0}, {1.0}, {1.0}, {0.0}};

    net.learn(inputs, labels);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
    }
}