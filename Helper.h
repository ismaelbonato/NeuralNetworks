#pragma once

#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <utility>

#include "training/FeedforwardTrainer.h"
#include "training/LayerParameterInitializer.h"
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

    DenseLayerRecipe config{};
    config.name = "Model";
    config.type = "DenseLayer";
    config.info = "info";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = inputs.at(0).size();
    config.outputSize = labels.at(0).size();
    config.expectedInputShape = {inputs.at(0).size()};
    config.expectedOutputShape = {labels.at(0).size()};

    auto skill = makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<UniformInitializer<Scalar>>(
             Scalar{-1.0},
             Scalar{1.0}),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
                     .intoSkill();

    Model net;
    net.addSkill(std::move(skill));
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

    DenseLayerRecipe config{};
    config.name = "Model";
    config.type = "DenseLayer";
    config.info = "info";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    auto skill = makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
                     .intoSkill();

    Model net;
    net.addSkill(std::move(skill));
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

    Batch labels = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //bart
                       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  //homer
                       {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  //marge
                       {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  //meg
                       {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  //grandpa
                       {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; //lisa

    auto col = inputs.at(0).size();

    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    DenseLayerRecipe config1{};
    config1.name = "Input";
    config1.type = "DenseLayer";
    config1.info = "info";
    config1.activation = activation;
    config1.inputSize = col;
    config1.outputSize = 32;
    config1.expectedInputShape = {col};
    config1.expectedOutputShape = {32};

    DenseLayerRecipe config2{};
    config2.name = "Hidden Layer";
    config2.type = "DenseLayer";
    config2.info = "info";
    config2.activation = activation;
    config2.inputSize = 32;
    config2.outputSize = 16;
    config2.expectedInputShape = {32};
    config2.expectedOutputShape = {16};

    DenseLayerRecipe config3{};
    config3.name = "Hidden Layer";
    config3.type = "DenseLayer";
    config3.info = "info";
    config3.activation = activation;
    config3.inputSize = 16;
    config3.outputSize = 8;
    config3.expectedInputShape = {16};
    config3.expectedOutputShape = {8};

    DenseLayerRecipe config4{};
    config4.name = "Output";
    config4.type = "DenseLayer";
    config4.info = "info";
    config4.activation = activation;
    config4.inputSize = 8;
    config4.outputSize = labels.size();
    config4.expectedInputShape = {8};
    config4.expectedOutputShape = {labels.size()};

    LayerParameterInitialization initialization{
        .weightInitializer = std::make_shared<UniformInitializer<Scalar>>(
            Scalar{-1.0},
            Scalar{1.0}),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()};
    auto skill1 = makeTrainableSkill<DenseLayer>(config1, initialization).intoSkill();
    auto skill2 = makeTrainableSkill<DenseLayer>(config2, initialization).intoSkill();
    auto skill3 = makeTrainableSkill<DenseLayer>(config3, initialization).intoSkill();
    auto skill4 = makeTrainableSkill<DenseLayer>(config4, initialization).intoSkill();
    

    Model net;
    net.addSkill(std::move(skill1));
    net.addSkill(std::move(skill2));
    net.addSkill(std::move(skill3));
    net.addSkill(std::move(skill4));
    FeedforwardTrainer trainer;

    trainer.learn(net, inputs, labels, Scalar{0.1f}, 100000);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        //std::cout << input << std::endl;
        std::cout << output << std::endl;
    }
}
