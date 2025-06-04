#ifndef HOPFIELD_H
#define HOPFIELD_H
#include "Model.h"

#include <vector>
#include <memory>
#include <cstddef> // for size_t

#include "Layer.h"
#include "HopfieldLayer.h"
#include "LearningRule.h"

class Hopfield: public Model
{
public:
    Hopfield(size_t n)
        : Model(std::make_unique<HebbianRule>(), n)
    {
        layers.emplace_back(std::make_unique<HopfieldLayer>(n, n));
    }

    // Constructor with custom learning rule
    Hopfield(std::unique_ptr<LearningRule> newRule, size_t n)
        : Model(std::move(newRule), n)
    {
        layers.emplace_back(std::make_unique<HopfieldLayer>(n, n));
    }

    ~Hopfield() override = default;
private:

};

#endif // HOPFIELD_H