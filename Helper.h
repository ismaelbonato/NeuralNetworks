#pragma once

#include "base/ActivationFunction.h"
#include "base/Layer.h"
#include "base/LayerFactory.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "base/Types.h"
#include "base/Model.h"
#include <opencv2/opencv.hpp>

#include <utility>

#include "training/FeedforwardTrainer.h"
#include "training/NaturalSelectionTrainer.h"
#include "training/PerceptronRuleTrainer.h"


inline Pattern png_to_bits(const std::string &filename)
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
                pixel > 128 ? Scalar{1.0f}
                            : Scalar{}); // Convert to binary representation
        }
    }
    return pattern;
}

inline void perceptronNetwork()
{
    Batch inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

    Batch labels = {
        {0.0}, // 0 AND 0
        {0.0}, // 0 AND 1
        {0.0}, // 1 AND 0
        {1.0}  // 1 AND 1
    };

    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize =inputs.at(0).size(),
        .outputSize = labels.at(0).size(),
        .name = "Model",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0}, Scalar{1.0}),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    auto l = makeLayer<DenseLayer>(config);

    Model net;
    net.addLayer(std::move(l));
    PerceptronRuleTrainer trainer;

    trainer.learn(net, inputs, labels, Scalar{0.1f}, 1000);

    std::cout << "Model Network trained!" << std::endl;
    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
    }
        
}


inline void perceptronNaturalSelection()
{
    Batch inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

    Batch labels = {
        {0.0}, // 0 AND 0
        {0.0}, // 0 AND 1
        {0.0}, // 1 AND 0
        {1.0}  // 1 AND 1
    };

    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 1,
        .name = "Model",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    auto l = makeLayer<DenseLayer>(config);

    Model net;
    net.addLayer(std::move(l));
    NaturalSelectionTrainer trainer;

    trainer.learn(net, inputs, labels, Scalar{0.1f}, 10000);

    std::cout << "Model Network trained!" << std::endl;
    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
    }
        
}

inline void feedforwardExperiment()
{
    Batch inputs;

    inputs.emplace_back(png_to_bits("../Misc/bart.png"));
    inputs.emplace_back(png_to_bits("../Misc/homer.png"));
    inputs.emplace_back(png_to_bits("../Misc/marge.png"));
    inputs.emplace_back(png_to_bits("../Misc/meg.png"));
    inputs.emplace_back(png_to_bits("../Misc/grandpa.png"));
    inputs.emplace_back(png_to_bits("../Misc/lisa.png"));

    Pattern p(png_to_bits("../Misc/meg.png"));

    Batch labels = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //bart
                       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  //homer
                       {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  //marge
                       {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  //meg
                       {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  //grandpa
                       {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; //lisa

    auto col = inputs.at(0).size();

    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    auto weightInitializer = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0}, Scalar{1.0});
    auto biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();

    LayerConfig config1{
        .learningRule = rule,
        .activation = activation,
        .inputSize = col,
        .outputSize = 32,
        .name = "Input",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = weightInitializer,
        .biasInitializer = biasInitializer,
    };

    LayerConfig config2{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 32,
        .outputSize = 16,
        .name = "Hidden Layer",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = weightInitializer,
        .biasInitializer = biasInitializer,
    };

    LayerConfig config3{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 16,
        .outputSize = 8,
        .name = "Hidden Layer",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = weightInitializer,
        .biasInitializer = biasInitializer,
    };
    
    LayerConfig config4{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 8,
        .outputSize = labels.size(),
        .name = "Output",
        .type = "DenseLayer",
        .info = "info",
        .weightInitializer = weightInitializer,
        .biasInitializer = biasInitializer,
    };

    auto l1 = makeLayer<DenseLayer>(config1);
    auto l2 = makeLayer<DenseLayer>(config2);
    auto l3 = makeLayer<DenseLayer>(config3);
    auto l4 = makeLayer<DenseLayer>(config4);
    

    Model net;
    net.addLayer(std::move(l1));
    net.addLayer(std::move(l2));
    net.addLayer(std::move(l3));
    net.addLayer(std::move(l4));
    FeedforwardTrainer trainer;

    trainer.learn(net, inputs, labels, Scalar{0.1f}, 100000);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        //std::cout << input << std::endl;
        std::cout << output << std::endl;
    }
}