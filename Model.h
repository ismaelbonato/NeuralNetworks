#ifndef MODEL_H
#define MODEL_H
#include "LearningRule.h"
#include "Layer.h" // Include the header file where Layer is defined
#include <vector>
#include <memory>
#include <cstddef> // Include for size_t

class Model
{
public:
    Model() = default;

    // Constructor that takes ownership of a LearningRule and sets the number of neurons
    Model(std::unique_ptr<LearningRule> rule, size_t n)
        : nNeurons(n), learningRule(std::move(rule)) {}

    virtual ~Model() = default;
protected:
    size_t nNeurons;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LearningRule> learningRule;

public:
    void setLearningRule(std::unique_ptr<LearningRule> rule) {
        learningRule = std::move(rule);
    }

    // Add a layer to the model
    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    // Remove a layer by index
    void removeLayer(size_t index) {
        if (index < layers.size()) {
            layers.erase(layers.begin() + static_cast<std::vector<std::unique_ptr<Layer>>::difference_type>(index));
        }
    }

    // Call the learning rule's learn method
    void learn(const std::vector<Pattern>& patterns) {
        if (!learningRule) return; 
        if (layers.empty()) return; 

        for (auto& layer : layers) { // Ensure 'auto&' for proper iteration
            layer->setWeights(learningRule->learn(patterns));
        }
    }
    
    Pattern forward(const Pattern& input) {
        if (!layers.empty()) {
            return layers.front()->forward(input);
        }
        return input; // or throw an exception if no layers exist
    }
    //virtual OutputType predict(const InputType& input) const = 0;

private:
    // You can add private members or methods if needed
};







#endif // MODEL_H