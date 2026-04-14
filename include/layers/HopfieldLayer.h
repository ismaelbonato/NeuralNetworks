#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;
    HopfieldLayer(const LayerConfig &newConfig);
    ~HopfieldLayer() override;

    std::shared_ptr<Layer> clone() const override;
    Pattern infer(const Pattern &input) const override;
    void updateWeights(const Pattern &pattern,
                       const Pattern &layerDelta,
                       Scalar learningRate = Scalar{1.0f}) override;
    void initWeights(Scalar value = Scalar{}) override;
    Pattern recall(const Pattern &input) const;
};
