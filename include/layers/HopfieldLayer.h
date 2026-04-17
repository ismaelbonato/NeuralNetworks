#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;
    ~HopfieldLayer() override;
    Pattern infer(const Pattern &input) const override;
    void updateWeights(const Pattern &pattern,
                       const Pattern &layerDelta,
                       Scalar learningRate = Scalar{1.0f}) override;
    Pattern recall(const Pattern &input) const;

protected:
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    explicit HopfieldLayer(const LayerConfig &newConfig);

    template<typename LayerType>
    friend std::unique_ptr<LayerType> makeLayer(const LayerConfig &config);
};
