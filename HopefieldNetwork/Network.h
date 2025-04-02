#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include <iostream>
#include <vector>


class Network
{
public:
    std::vector<Neuron> neurons;
    std::vector<int> output;
    int threshld(int);
    void activation(Pattern &patrn);
    Network(Weights &a, Weights &b, Weights &c, Weights &d);
};

#endif