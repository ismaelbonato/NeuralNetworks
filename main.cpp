#include "Helper.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <memory>

#include "base/Layer.h"
#include "base/LearningRule.h"
#include "networks/FeedForward.h"
#include "networks/Hopfield.h"
#include "networks/Perceptron.h"

void feedforwardManual()
{
    std::cout << "Manual Feedforward Network" << std::endl;

    std::vector<size_t> layerSizes = {2, 4, 2, 1};
    //FeedforwardNetwork net(layerSizes);
    FeedforwardNetwork net;

    net.addLayer(std::make_unique<FeedforwardLayer>(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        2,
        4));

    net.addLayer(std::make_unique<FeedforwardLayer>(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        4,
        2));
    net.addLayer(std::make_unique<FeedforwardLayer>(
        std::make_shared<SGDRule<Scalar>>(),
        std::make_shared<SigmoidActivation<Scalar>>(),
        2,
        1));

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
    std::cout << "Manual Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    feedforwardManual();
    std::cout << std::endl;

    return 0;
}