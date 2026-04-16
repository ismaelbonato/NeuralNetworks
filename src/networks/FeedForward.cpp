#include "networks/FeedForward.h"

Feedforward::Feedforward() = default;

Feedforward::Feedforward(Layers &newlayers)
    : GradientBaseModel(newlayers)
{}

Feedforward::Feedforward(const std::initializer_list<std::shared_ptr<Layer>> &newlayers)
    : GradientBaseModel(newlayers)
{}

Feedforward::~Feedforward() = default;
