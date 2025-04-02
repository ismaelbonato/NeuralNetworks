#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

using Weights = std::array<int, 4>;
using Pattern = std::array<int, 4>;

class Neuron 
{
protected:
    int activation;
    friend class Network;

public:
    std::vector<int> weights;
    Neuron(){};
    Neuron(Weights &ws);
    int act(const int m, Pattern &x);
};

#endif