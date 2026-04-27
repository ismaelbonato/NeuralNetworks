#pragma once

#include "base/Layer.h"

class ConvolutionalLayer : public TrainableLayer
{
public:
    ConvolutionalLayer() = delete;
    ~ConvolutionalLayer() override;

    Pattern preActivation(const Pattern &input) const override;
    Pattern backwardPass(const Pattern &layerDelta,
                         const Pattern &layerInput) const override;

    void updateWeights(const Pattern &prev_activations,
                       const Pattern &layerDelta,
                       Scalar learningRate) override;

protected:
    ConvolutionalLayerConfig convolutionalConfig;

    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;

private:
    explicit ConvolutionalLayer(const ConvolutionalLayerConfig &newConfig);

    template<typename LayerType, typename ConfigType>
    friend std::unique_ptr<LayerType> makeLayer(const ConfigType &config);
};
