#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete;
    DenseLayer(const LayerConfig &newConfig);
    ~DenseLayer() override;

    std::shared_ptr<Layer> clone() const override;
    void initWeights(Scalar value = Scalar{}) override;
};
