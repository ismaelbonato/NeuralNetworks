#pragma once

#include "base/Layer.h"
#include "base/Types.h"

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;
    explicit HopfieldLayer(const HopfieldLayerRecipe &newRecipe);
    ~HopfieldLayer() override;
    const Pattern &getWeights() const;
    const Pattern &getBiases() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void setWeights(const Pattern &newWeights);
    void setBiases(const Pattern &newBiases);
    bool isInitialized() const;
    void requireInitialized() const;
    Pattern recall(const Pattern &input) const;

protected:
    Pattern forward(const Pattern &input) const override;

private:
    Pattern weights;
    Pattern biases;

    Shape expectedWeightShape() const;
    Shape expectedBiasShape() const;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input) const;
    Pattern activate(const Pattern &values) const;
};
