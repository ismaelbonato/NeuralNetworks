#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>

using Pattern = std::vector<int>;
using Weights = std::vector<double>;
using RowOfWeights = std::vector<Weights>;


// Convert 0/1 pattern to -1/+1
constexpr inline int bin_to_bipolar(int x) {
    return 2 * x - 1;
}

// Convert 0/1 pattern to -1/+1
constexpr inline double bipolarToBin(double x) {
    return (x + 1.0) / 2.0;
}

class Neuron
{
protected:
    double activation;
    friend class Network;

public:
    Weights weights;
    Neuron() = default;
    Neuron(const Weights &ws);
    double act(const Pattern &pattern);
};

#endif