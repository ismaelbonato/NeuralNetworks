#pragma once

#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/Parameters.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace nn {

struct LayerRecipe
{
    virtual ~LayerRecipe() = default;

    std::string name;
    std::string type;
    std::string info;
    std::shared_ptr<ActivationFunction<Scalar>> activation;

    virtual Shape getInputShape() const = 0;
    virtual Shape getOutputShape() const = 0;
    virtual void validateRecipe() const;
};

class Layer
{
protected:
    std::unique_ptr<LayerRecipe> recipe;
    std::optional<Parameters> ownedParameters;

    void requireInputShape(const Pattern &input) const;
    virtual Pattern forward(const Pattern &input) const;
    virtual Pattern forward(const Pattern &input,
                            const Parameters &parameters) const;
    virtual Pattern weightedInput(const Pattern &input,
                                  const Parameters &parameters) const;
    Pattern activate(const Pattern &values) const;

public:
    Layer() = delete;
    explicit Layer(std::unique_ptr<LayerRecipe> newRecipe);

    virtual ~Layer();

    Shape getInputShape() const;
    Shape getOutputShape() const;
    const std::shared_ptr<ActivationFunction<Scalar>> &getActivation() const;
    const std::string &getName() const;
    const std::string &getType() const;
    const std::string &getInfo() const;
    bool usesParameters() const;
    virtual Shape expectedWeightShape() const;
    virtual Shape expectedBiasShape() const;
    void requireValidParameters(const Parameters &parameters) const;
    std::optional<Parameters> parameters() const;
    const Parameters &getParameters() const;
    void setParameters(const Parameters &parameters);
    void requireParameters() const;

    Pattern infer(const Pattern &input) const;
};

} // namespace nn
