#include "training/LayerParameterInitializer.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"

#include <functional>
#include <optional>
#include <typeinfo>

namespace
{
template<typename LayerType>
std::optional<std::reference_wrapper<LayerType>> layerAs(Layer &layer)
{
    try {
        return std::ref(dynamic_cast<LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}
} // namespace

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

void initializeDenseLayer(
    DenseLayer &layer,
    const LayerParameterInitialization &initialization)
{
    if (layer.getWeights().empty()) {
        layer.setWeights(initializedParameter({layer.getOutputSize(),
                                               layer.getInputSize()},
                                              initialization.weightInitializer));
    }

    if (layer.getBiases().empty()) {
        layer.setBiases(initializedParameter({layer.getOutputSize()},
                                             initialization.biasInitializer));
    }
}

void initializeConvolutionalLayer(
    ConvolutionalLayer &layer,
    const LayerParameterInitialization &initialization)
{
    const auto &recipe = layer.getConvolutionalRecipe();
    if (layer.getWeights().empty()) {
        layer.setWeights(initializedParameter({recipe.outputChannels,
                                               recipe.inputChannels,
                                               recipe.kernelSize},
                                              initialization.weightInitializer));
    }

    if (layer.getBiases().empty()) {
        layer.setBiases(initializedParameter({recipe.outputChannels},
                                             initialization.biasInitializer));
    }
}

void initializeHopfieldLayer(
    HopfieldLayer &layer,
    const LayerParameterInitialization &initialization)
{
    if (layer.getWeights().empty()) {
        layer.setWeights(initializedParameter({layer.getOutputSize(),
                                               layer.getInputSize()},
                                              initialization.weightInitializer));
    }

    if (layer.getBiases().empty()) {
        layer.setBiases({});
    }
}

void initializeLayerParameters(
    Layer &layer,
    const LayerParameterInitialization &initialization)
{
    if (auto dense = layerAs<DenseLayer>(layer)) {
        initializeDenseLayer(dense->get(), initialization);
    } else if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
        initializeConvolutionalLayer(convolutional->get(), initialization);
    } else if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
        initializeHopfieldLayer(hopfield->get(), initialization);
    }
}

void initializeSkillParameters(
    Skill &skill,
    const LayerParameterInitialization &initialization)
{
    initializeLayerParameters(skill.layer(), initialization);
    skill.adoptLayerParameters();
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
