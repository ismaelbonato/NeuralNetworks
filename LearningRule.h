#ifndef LEARNINGRULE_H
#define LEARNINGRULE_H

#include <vector>
#include <memory>
#include <cstddef> // for size_t
#include <iostream>
#include "Layer.h"

class LearningRule;

class LearningRule
{
public:
    LearningRule() = default;

    virtual ~LearningRule() = default;

    // Add any common methods or properties for all learning rules here
    virtual std::vector<Pattern> learn(const std::vector<Pattern> &patterns) = 0; // Pure virtual function for applying the learning rule
    //virtual void reset() = 0; // Pure virtual function for resetting the learning rule state
};

class HebbianRule : public LearningRule
{
public:
    HebbianRule() = default;
    ~HebbianRule() override = default;

    // Apply the Hebbian learning rule to update weights
    std::vector<Pattern> learn(const std::vector<Pattern> &patterns) override
    {
        size_t n = patterns[0].size();
        std::vector<Pattern> weights;

        weights.assign(n, std::vector<double>(n, 0.0));
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
};

#endif // LEARNINGRULE_H