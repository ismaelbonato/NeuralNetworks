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

    DenseLayerConfig config;
    config.name = "Model";
    config.type = "DenseLayer";
    config.info = "info";
    config.learningRule = std::make_shared<PerceptronRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer =
        std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0}, Scalar{1.0});
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.inputSize = inputs.at(0).size();
    config.outputSize = labels.at(0).size();
    config.expectedInputShape = {inputs.at(0).size()};
    config.expectedOutputShape = {labels.at(0).size()};

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

    DenseLayerConfig config;
    config.name = "Model";
    config.type = "DenseLayer";
    config.info = "info";
    config.learningRule = std::make_shared<PerceptronRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

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

    DenseLayerConfig config1;
    config1.name = "Input";
    config1.type = "DenseLayer";
    config1.info = "info";
    config1.learningRule = rule;
    config1.activation = activation;
    config1.weightInitializer = weightInitializer;
    config1.biasInitializer = biasInitializer;
    config1.inputSize = col;
    config1.outputSize = 32;
    config1.expectedInputShape = {col};
    config1.expectedOutputShape = {32};

    DenseLayerConfig config2;
    config2.name = "Hidden Layer";
    config2.type = "DenseLayer";
    config2.info = "info";
    config2.learningRule = rule;
    config2.activation = activation;
    config2.weightInitializer = weightInitializer;
    config2.biasInitializer = biasInitializer;
    config2.inputSize = 32;
    config2.outputSize = 16;
    config2.expectedInputShape = {32};
    config2.expectedOutputShape = {16};

    DenseLayerConfig config3;
    config3.name = "Hidden Layer";
    config3.type = "DenseLayer";
    config3.info = "info";
    config3.learningRule = rule;
    config3.activation = activation;
    config3.weightInitializer = weightInitializer;
    config3.biasInitializer = biasInitializer;
    config3.inputSize = 16;
    config3.outputSize = 8;
    config3.expectedInputShape = {16};
    config3.expectedOutputShape = {8};

    DenseLayerConfig config4;
    config4.name = "Output";
    config4.type = "DenseLayer";
    config4.info = "info";
    config4.learningRule = rule;
    config4.activation = activation;
    config4.weightInitializer = weightInitializer;
    config4.biasInitializer = biasInitializer;
    config4.inputSize = 8;
    config4.outputSize = labels.size();
    config4.expectedInputShape = {8};
    config4.expectedOutputShape = {labels.size()};

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
