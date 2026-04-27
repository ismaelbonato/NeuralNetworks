#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class HopfieldLayer : public TrainableLayer
{
public:
    HopfieldLayer() = delete;
    ~HopfieldLayer() override;
    void updateWeights(const Pattern &pattern,
                       const Pattern &layerDelta,
                       Scalar learningRate = Scalar{1.0f}) override;
    Pattern recall(const Pattern &input) const;

protected:
    Pattern forward(const Pattern &input) const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    explicit HopfieldLayer(const HopfieldLayerConfig &newConfig);

    template<typename LayerType, typename ConfigType>
    friend std::unique_ptr<LayerType> makeLayer(const ConfigType &config);
};
