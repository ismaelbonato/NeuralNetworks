#pragma once

#include "Perceptron.h"
#include "base/Types.h"
#include <cstddef>
#include <map>

class PerceptronNaturalSelection : public Perceptron
{
public:
    PerceptronNaturalSelection()
        = default; // No layers by default, can be extended later

    PerceptronNaturalSelection(const std::shared_ptr<Layer> &newLayer)
        : Perceptron({newLayer})
    {}

    ~PerceptronNaturalSelection() override = default;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override
    {

        Layers ls;
        ls.push_back(layers.front()->clone());
        ls.push_back(layers.front()->clone());
        ls.push_back(layers.front()->clone());
        ls.push_back(layers.front()->clone());

        size_t bestIdx = 0;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            Patterns ret(4);

            for (size_t i = 0; i < inputs.size(); ++i) {
                
                ret[0].push_back(ls[0]->infer(inputs[i]).front());
                ret[1].push_back(ls[1]->infer(inputs[i]).front());
                ret[2].push_back(ls[2]->infer(inputs[i]).front());
                ret[3].push_back(ls[3]->infer(inputs[i]).front());
            }

            size_t bestIdx = findClosestPerceptron(ret, labels);

            ls[0]->naturalUpdateWeights(*ls[bestIdx]);
            ls[1]->naturalUpdateWeights(*ls[bestIdx]);
            ls[2]->naturalUpdateWeights(*ls[bestIdx]);
            ls[3]->naturalUpdateWeights(*ls[bestIdx]);


        }
        layers.front()->naturalUpdateWeights(*ls[bestIdx]);

    }
    size_t findClosestPerceptron(const Patterns& ret, const Patterns& labels) const
    {
        size_t bestIdx = 0;
        Scalar bestError = std::numeric_limits<Scalar>::max();

        for (size_t perceptronIdx = 0; perceptronIdx < ret.size(); ++perceptronIdx) {
            Scalar totalError = 0.0f;
            for (size_t sampleIdx = 0; sampleIdx < ret[perceptronIdx].size(); ++sampleIdx) {
                Scalar diff = ret[perceptronIdx][sampleIdx] - labels[sampleIdx].front();
                totalError += diff * diff;
            }
            if (totalError < bestError) {
                bestError = totalError;
                bestIdx = perceptronIdx;
            }
        }
        return bestIdx;
    }

};