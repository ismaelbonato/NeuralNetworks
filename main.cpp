#include "Helper.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <memory>

#include "base/Types.h"
#include "base/Layer.h"
#include "base/LearningRule.h"
#include "networks/FeedForward.h"
#include "networks/Hopfield.h"
#include "networks/Perceptron.h"
#include "networks/DenseLayer.h"

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

    Patterns labels = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, //bart
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, //homer
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, //marge
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, //meg
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, //grandpa
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};  //lisa

    auto col = inputs.at(0).size();
    
    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();

    DenseLayer layer1(rule, activation, col, 32);
    DenseLayer layer2(rule, activation, 32, 16);
    DenseLayer layer3(rule, activation, 16, 6);

    Feedforward net(layer1, layer2, layer3);

    net.learn(inputs, labels);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        //std::cout << input << std::endl;
        std::cout << output << std::endl;;
    }
}

int main()
{
    std::cout << "Hello, Neural Networks!" << std::endl;

    std::cout << "==========================" << std::endl;
    std::cout << "Perceptron Network" << std::endl;
    std::cout << "==========================" << std::endl;
    perceptronNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Hopfield Network" << std::endl;
    std::cout << "==========================" << std::endl;
    hopfieldNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    feedForwardNetwork();
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "DependencyInversion Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << std::endl;

    //feedforwardExperiment();

    return 0;
}