#include "Neuron.h"
#include <array>

Neuron::Neuron(Weights &ws)
{
    for (auto w : ws) {
        weights.emplace_back(w);
    }
}

int Neuron::act(const int m, Pattern &x)
{
    int i;
    int a = 0;

    for (int idx = 0; idx < m; ++idx) {
        a += x.at(idx) * weights.at(idx);
    }

    return a;
}
