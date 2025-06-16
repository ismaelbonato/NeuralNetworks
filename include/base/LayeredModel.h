#pragma once

#include "Model.h"
#include "Types.h"

class LayeredModel : public Model
{
public:
    LayeredModel() = default;
    virtual ~LayeredModel() = default;    

    Layers& getLayers()
    {
        return layers;
    }

    const Layers& getLayers() const;

    size_t numLayers() const
    {
        return layers.size();
    }

    // Add a layer to the model
    virtual void addLayer(std::unique_ptr<Layer> layer)
    { 
        layers.push_back(std::move(layer));
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