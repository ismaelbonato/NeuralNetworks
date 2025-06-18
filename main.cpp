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

void feedforward_DependencyInversion()
{
    std::cout << "Manual Feedforward Network" << std::endl;
    
    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();

    DenseLayer layer1(rule, activation, 2, 4);
    DenseLayer layer2(rule, activation, 4, 2);
    DenseLayer layer3(rule, activation, 2, 1);
    
    Layers layers;
    
    layers.push_back(layer1.clone());
    layers.push_back(layer2.clone());
    layers.push_back(layer3.clone());

    FeedforwardNetwork net(std::move(layers));

    Patterns inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    Patterns labels = {{0.0}, {1.0}, {1.0}, {0.0}};

    net.learn(inputs, labels);

    for (const auto &input : inputs) {
        Pattern output = net.infer(input);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
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
    feedforward_DependencyInversion();
    std::cout << std::endl;

    return 0;
}