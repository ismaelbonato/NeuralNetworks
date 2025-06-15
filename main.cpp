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

void feedForwardNetwork()
{
    std::vector<Pattern> inputs;

    inputs.emplace_back(png_to_bits("../Misc/bart.png"));
    inputs.emplace_back(png_to_bits("../Misc/homer.png"));
    inputs.emplace_back(png_to_bits("../Misc/marge.png"));
    inputs.emplace_back(png_to_bits("../Misc/meg.png"));
    inputs.emplace_back(png_to_bits("../Misc/grandpa.png"));
    inputs.emplace_back(png_to_bits("../Misc/lisa.png"));

    Pattern p(png_to_bits("../Misc/meg.png"));

    std::vector<Pattern> labels = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, //bart
        {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, //homer
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, //marge
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, //meg
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, //grandpa
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};  //lisa

/*
std::vector<Pattern> inputs = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};

std::vector<Pattern> labels = {
    {0.0f}, // 0 XOR 0 = 0
    {1.0f}, // 0 XOR 1 = 1
    {1.0f}, // 1 XOR 0 = 1
    {0.0f}  // 1 XOR 1 = 0
};
*/
    size_t N = inputs.at(0).size();
    size_t O = labels.at(0).size();
    
    std::vector<size_t> layers = { N, 16, 8, O};
    FeedforwardNetwork mlp(layers);

    float learningRate = 0.1f;
    size_t epochs = 10000;
    mlp.learn(inputs, labels, learningRate, epochs);

    size_t index = 1;
    for (const auto &input : inputs) {
        Pattern output = mlp.infer(input);
        std::cout << "Input: " << index++ << " - ";
        std::cout << "Output: " << "{ ";
        for (auto &i : output)
        {
            std::cout << i << ", ";
        }
        std::cout << "}" << std::endl;
    }
}

int main()
{
    feedForwardNetwork();
    //perceptronNetwork();
    //hopfieldNetwork();
    
    return 0;
}