#ifndef LEARNINGRULE_H
#define LEARNINGRULE_H

#include <cstddef> // for size_t
#include <iostream>
#include <memory>
#include <vector>

using Pattern = std::vector<double>;

class LearningRule
{
public:
    LearningRule() = default;

    virtual ~LearningRule() = default;

    virtual std::vector<Pattern> learn(const std::vector<Pattern> &patterns) = 0;
    virtual std::vector<Pattern> learn(const std::vector<Pattern> &inputs,
                                       const std::vector<Pattern> &labels)
        = 0;
};

class HebbianRule : public LearningRule
{
public:
    HebbianRule() = default;
    ~HebbianRule() override = default;

    using LearningRule::learn;

    // Apply the Hebbian learning rule to update weights
    std::vector<Pattern> learn(const std::vector<Pattern> &patterns) override
    {
        size_t n = patterns[0].size();
        std::vector<Pattern> weights;

        weights.assign(n, Pattern(n, 0.0));
        for (const auto &pattern : patterns) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    if (i != j) {
                        weights[i][j] += pattern[i] * pattern[j];
                    }
                }
            }
        }
        // Optionally normalize weights here
        return weights; // Return the updated weights
    }
    std::vector<Pattern> learn(const std::vector<Pattern> &,
                               const std::vector<Pattern> &) override
    {
        return {};
    }
};

class PerceptronRule : public LearningRule
{
public:
    PerceptronRule(double rate = 0.1)
        : learningRate(rate)
    {}
    ~PerceptronRule() override = default;

    using LearningRule::learn;

    // Supervised perceptron learning rule
    std::vector<Pattern> learn(const std::vector<Pattern> &inputs,
                               const std::vector<Pattern> &labels) override
    {
        size_t nSamples = inputs.size();
        size_t nInputs = inputs[0].size();
        size_t nOutputs = labels[0].size();

        // Initialize weights to zero
        std::vector<Pattern> weights(nOutputs, Pattern(nInputs, 0.0));

        // Simple online perceptron learning (single epoch)
        for (size_t sample = 0; sample < nSamples; ++sample) {
            const Pattern &input = inputs[sample];
            const Pattern &target = labels[sample];

            for (size_t out = 0; out < nOutputs; ++out) {
                // Compute weighted sum
                double sum = 0.0;
                for (size_t in = 0; in < nInputs; ++in) {
                    sum += weights[out][in] * input[in];
                }
                // Step activation
                double output = sum > 0 ? 1.0 : 0.0;
                // Update weights
                for (size_t in = 0; in < nInputs; ++in) {
                    weights[out][in] += learningRate * (target[out] - output) * input[in];
                }
            }
        }
        return weights;
    }

    // For unsupervised, just return empty (not used)
    std::vector<Pattern> learn(const std::vector<Pattern> &) override { return {}; }

private:
    double learningRate;
};

#endif // LEARNINGRULE_H