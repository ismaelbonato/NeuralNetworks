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
    Pattern forward(const Pattern &input,
                    const LayerParameters &parameters) const override;

private:
    Pattern weights;
    Pattern biases;

    bool usesParameters() const override;
    Shape expectedWeightShape() const override;
    Shape expectedBiasShape() const override;
    bool hasWeights() const;
    bool hasBias() const;
    Pattern weightedInput(const Pattern &input) const;
    Pattern weightedInput(const Pattern &input,
                          const LayerParameters &parameters) const;
    Pattern activate(const Pattern &values) const;
    Pattern recall(const Pattern &input,
                   const LayerParameters &parameters) const;
    bool isInitialized(const LayerParameters &parameters) const override;
    void requireInitialized(const LayerParameters &parameters) const override;
};
