#include "Helper.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <memory>

#include "base/Layer.h"
#include "base/LearningRule.h"
#include "base/Types.h"
#include "layers/DenseLayer.h"
#include "networks/FeedForward.h"
#include "networks/Hopfield.h"
#include "networks/Perceptron.h"

#include "base/Tensor.h"

void feedforwardExperiment()
{
    Patterns inputs;

    inputs.emplace_back(png_to_bits("../Misc/bart.png"));
    inputs.emplace_back(png_to_bits("../Misc/homer.png"));
    inputs.emplace_back(png_to_bits("../Misc/marge.png"));
    inputs.emplace_back(png_to_bits("../Misc/meg.png"));
    inputs.emplace_back(png_to_bits("../Misc/grandpa.png"));
    inputs.emplace_back(png_to_bits("../Misc/lisa.png"));

    Pattern p(png_to_bits("../Misc/meg.png"));

    Patterns labels = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //bart
                       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},  //homer
                       {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},  //marge
                       {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},  //meg
                       {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},  //grandpa
                       {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; //lisa

    auto col = inputs.at(0).size();

    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();

    LayerConfig config1{
        .learningRule = rule,
        .activation = activation,
        .inputSize = col,
        .outputSize = 32,
        .name = "Input",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    LayerConfig config2{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 32,
        .outputSize = 16,
        .name = "Hidden Layer",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    LayerConfig config3{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 16,
        .outputSize = 8,
        .name = "Hidden Layer",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };
    
    LayerConfig config4{
        .learningRule = rule,
        .activation = activation,
        .inputSize = 8,
        .outputSize = labels.size(),
        .name = "Output",
        .type = "DenseLayer",
        .info = "info",
        .useBias = true
    };

    auto l1 = std::make_shared<DenseLayer>(config1);
    auto l2 = std::make_shared<DenseLayer>(config2);
    auto l3 = std::make_shared<DenseLayer>(config3);
    auto l4 = std::make_shared<DenseLayer>(config4);

    Feedforward net({l1, l2, l3, l4});

    net.learn(inputs, labels);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        //std::cout << input << std::endl;
        std::cout << output << std::endl;
        ;
    }
}

int main()
{
    std::cout << "Hello, Neural Networks!" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Perceptron Network" << std::endl;
    std::cout << "==========================" << std::endl;
    //perceptronNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Hopfield Network" << std::endl;
    std::cout << "==========================" << std::endl;
    //hopfieldNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    //feedForwardNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "DependencyInversion Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << std::endl;

    feedforwardExperiment();


    
    return 0;
}