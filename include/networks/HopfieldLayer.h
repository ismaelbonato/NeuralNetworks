#ifndef HOPFIELDLAYER_H
#define HOPFIELDLAYER_H
#include "base/Layer.h"

#include <stdexcept>
#include <vector>

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = default;

    HopfieldLayer(size_t in, size_t out)
        : Layer(in, out)
    {
        weights.resize(in, std::vector<double>(out, 0.0));
    }

    HopfieldLayer(std::unique_ptr<LearningRule> newRule, size_t in, size_t out)
        : Layer(std::move(newRule), in, out)
    {
        weights.resize(in, std::vector<double>(out, 0.0));
    }

    ~HopfieldLayer() override = default;

    int activation(double value) const override
    {
        return (value >= 0) ? 1 : -1; // Bipolar activation function
    }

    // Overload forward: update until convergence
    Pattern forward(const Pattern &input) const override
    {
        Pattern state = input;
        Pattern prev_state;
        do {
            prev_state = state;
            // Asynchronous update: update each neuron based on current state
            for (size_t i = 0; i < state.size(); ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < state.size(); ++j) {
                    if (i != j) {
                        sum += weights[i][j] * state[j];
                    }
                }
                state[i] = activation(sum);
            }
        } while (state != prev_state); // Repeat until state does not change
        return state;
    }
};

#endif // HOPFIELDLAYER_H