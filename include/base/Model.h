#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace nn {

class Model
{
protected:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) noexcept;
    Model &operator=(Model &&) noexcept;
    virtual ~Model();

    Layer &addLayer(std::unique_ptr<Layer> layer);

    Layer &getLayer(size_t index);
    const Layer &getLayer(size_t index) const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);

    static Model loadFromFile(const std::string &file);
    void saveToFile(const std::string &file) const;
};

} // namespace nn
