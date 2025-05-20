#include "Neuron.h"
#include <array>

Neuron::Neuron(const Weights &ws)
{
    weights.reserve(ws.size());

    for (auto &w : ws) {
        weights.emplace_back(w);
    }
    //std::cout << "amount of weights: " << weights.size() << std::endl;
}

double Neuron::act(const Pattern &pattern)
{
    double acc = 0.0;
    for (size_t idx = 0; idx < pattern.size(); ++idx) {
        acc += pattern.at(idx) * weights.at(idx);
    }

    return acc;
}
