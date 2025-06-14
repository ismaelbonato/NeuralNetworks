#pragma once
#include "base/Layer.h"

class FeedforwardLayer : public Layer
{

public:
    FeedforwardLayer(const std::shared_ptr<LearningRule> &rule, size_t in, size_t out)
        : Layer(rule, in, out)
    {
    }

    ~FeedforwardLayer() override = default;


    float activation(float value) const override
    {
        return 1.0f / (1.0f + std::exp(-value));
    }

};