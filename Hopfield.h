#ifndef HOPFIELD_H
#define HOPFIELD_H
#include "Model.h"

#include <cstddef> // for size_t
#include <memory>
#include <vector>

#include "HopfieldLayer.h"
#include "Layer.h"
#include "LearningRule.h"

class Hopfield : public Model
{
public:
    Hopfield(size_t n)
        : Model(n)
    {
        layers.emplace_back(
            std::make_unique<HopfieldLayer>(std::move(std::make_unique<HebbianRule>()), n, n));
    }

    // Constructor with custom learning rule
    Hopfield(std::unique_ptr<LearningRule> newRule, size_t n)
        : Model(n)
    {
        layers.emplace_back(std::make_unique<HopfieldLayer>(std::move(newRule), n, n));
    }

    ~Hopfield() override = default;

private:
};

#endif // HOPFIELD_H