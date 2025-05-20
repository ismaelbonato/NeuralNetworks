#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include <array>
#include <iostream>
#include <vector>

class Network
{
public:
    Network() = default;

    Network(const RowOfWeights &row);
    void activation(const Pattern &pattern);

    RowOfWeights &hebbianLearning(const std::vector<Pattern> &patterns);
    void updateNeurons(const RowOfWeights &rw);

    void printWeights();
    void printOutput();
    void printPattern(const Pattern &p);

private:
    RowOfWeights rowOfWeights;

    Pattern output;

    std::vector<Neuron> neurons;
    int threshld(double);
};

#endif