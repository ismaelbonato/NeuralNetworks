#pragma once

#include "Model.h"
#include "Types.h"

class LayeredModel : public Model
{
public:
    LayeredModel() = default;

    LayeredModel(Layers &newLayers)
        : Model(newLayers)
    {}

    LayeredModel(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
        : Model(newLayers)
    {}

    virtual ~LayeredModel() = default;

    inline Layers::value_type &getLayer(size_t index)
    {
        if (index >= layers.size()) {
            throw std::out_of_range("Layer index out of range.");
        }
        return layers[index];
    }

    inline Layers &getLayers() { return layers; }

    inline const Layers &getLayers() const { return layers; }

    size_t numLayers() const { return layers.size(); }

    virtual void addLayers(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
    {
        for (auto &layer : newLayers)
        {
            layers.push_back(layer);    
        }
    }

    // Remove a layer by index
    virtual void removeLayer(size_t index)
    {
        if (index < layers.size()) {
            layers.erase(
                layers.begin()
                + static_cast<
                    std::vector<std::shared_ptr<Layer>>::difference_type>(
                    index));
        }
    }

    Pattern infer(const Pattern &input) override
    {
        if (layers.empty()) {
            throw std::runtime_error(
                "No layers exist in the model to perform inference.");
        }

        Pattern output = input;
        for (auto &layer : layers) {
            output = layer->infer(output);
        }
        return output;
    }

protected:
};