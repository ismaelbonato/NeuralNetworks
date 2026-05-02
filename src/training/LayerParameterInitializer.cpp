#include "training/LayerParameterInitializer.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"

Pattern initializedParameter(
    const Shape &shape,
    const std::shared_ptr<Initializer<Scalar>> &initializer,
    Scalar fallbackValue = Scalar{})
{
    Pattern parameter = Pattern::withShape(shape, fallbackValue);

    if (initializer) {
        initializer->fill(parameter);
    }

    return parameter;
}

LayerParameters initializedParametersFor(
    const Layer &layer,
    const LayerParameters &currentParameters,
    const LayerParameterInitialization &initialization)
{
    LayerParameters parameters = currentParameters;

    const Shape weightShape = layer.expectedWeightShape();
    if (parameters.weights.empty() && !weightShape.empty()) {
        parameters.weights = initializedParameter(weightShape,
                                                  initialization.weightInitializer);
    }

    const Shape biasShape = layer.expectedBiasShape();
    if (parameters.biases.empty() && !biasShape.empty()) {
        parameters.biases = initializedParameter(biasShape,
                                                initialization.biasInitializer);
    }

    layer.requireInitialized(parameters);
    return parameters;
}

void initializeSkillParameters(
    Skill &skill,
    const LayerParameterInitialization &initialization)
{
    if (!skill.hasParameters()) {
        return;
    }

    skill.setParameters(initializedParametersFor(
        skill.layer(),
        skill.parameters().value_or(LayerParameters{}),
        initialization));
}

void initializeModelParameters(
    Model &network,
    const LayerParameterInitialization &initialization)
{
    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        initializeSkillParameters(network.getSkill(layerIndex),
                                  initialization);
    }
}
