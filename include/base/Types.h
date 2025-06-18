#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <cstddef> // for size_t
#include <initializer_list>

template<typename T>
class Tensor;

class Layer; // Forward declaration

using Scalar = float;
using Pattern = Tensor<Scalar>;
using Patterns = Tensor<Pattern>;

//using Pattern = std::vector<Scalar>;
//using Patterns = std::vector<Pattern>;
//using Matrix = std::vector<std::vector<Scalar>>;
//using MatrixPattern = std::vector<Pattern>;

//using PatternPair = std::pair<Pattern, Pattern>;

using Layers = std::vector<std::unique_ptr<Layer>>;

/*
class LayerManager {
public:
    LayerManager() = default;

    // Move-enabled initializer list constructor
    LayerManager(std::initializer_list<std::unique_ptr<Layer>> init) {
        for (auto& ptr : init) {
            layers_.push_back(std::move(const_cast<std::unique_ptr<Layer>&>(ptr)));
        }
    }

    void push_back(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer));
    }

    void add_clone(const Layer& layer) {
        layers_.push_back(layer.clone());
    }

    std::unique_ptr<Layer>& operator[](size_t idx) { return layers_[idx]; }
    const std::unique_ptr<Layer>& operator[](size_t idx) const { return layers_[idx]; }
    size_t size() const { return layers_.size(); }
    auto begin() { return layers_.begin(); }
    auto end() { return layers_.end(); }
    auto begin() const { return layers_.begin(); }
    auto end() const { return layers_.end(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
*/