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

void feedforward_DependencyInversion()
{
    std::cout << "Manual Feedforward Network" << std::endl;
    
    DenseLayer layer1(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        2, 4);

    DenseLayer layer2(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        4, 2);
    DenseLayer layer3(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        2, 1);  
    
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
        print(input);
        print(output);
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