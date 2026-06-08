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
#include <string_view>
#include <variant>
#include <vector>

namespace nn {

using LayerFieldValue = std::variant<size_t, std::string>;

struct LayerField
{
    std::string name;
    LayerFieldValue value;
};

struct LayerSnapshot
{
    std::string name;
    std::string type;
    std::string info;
    std::string activation;

    std::vector<size_t> inputShape;
    std::vector<size_t> outputShape;
    std::vector<LayerField> fields;
    std::optional<Parameters> parameters;
};

struct LayerRecipe
{
    virtual ~LayerRecipe() = default;

    std::string name;
    std::string type;
    std::string info;
    std::shared_ptr<ActivationFunction<Scalar>> activation;

    Shape inputShape;
    Shape outputShape;

    virtual const Shape &getInputShape() const;
    virtual const Shape &getOutputShape() const;
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

    virtual LayerSnapshot snapshot() const;

    const Shape &getInputShape() const;
    const Shape &getOutputShape() const;
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
