#include "base/Model.h"

#include <stdexcept>
#include <utility>

Model::Model() = default;

Model::~Model() = default;

Skill &Model::addSkill(Skill skill)
{
    skills.push_back(std::move(skill));
    return skills.back();
}

void Model::removeLayer(size_t index)
{
    if (index < skills.size()) {
        skills.erase(skills.begin()
                     + static_cast<Skills::difference_type>(index));
    }
}

Layer &Model::getLayer(size_t index)
{
    return getSkill(index).layer();
}

const Layer &Model::getLayer(size_t index) const
{
    return getSkill(index).layer();
}

Skill &Model::getSkill(size_t index)
{
    if (index >= skills.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return skills.at(index);
}

const Skill &Model::getSkill(size_t index) const
{
    if (index >= skills.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return skills.at(index);
}

const Skills &Model::getSkills() const
{
    return skills;
}

std::vector<std::reference_wrapper<Layer>> Model::getLayers()
{
    std::vector<std::reference_wrapper<Layer>> layers;
    layers.reserve(skills.size());
    for (auto &skill : skills) {
        layers.push_back(std::ref(skill.layer()));
    }
    return layers;
}

std::vector<std::reference_wrapper<const Layer>> Model::getLayers() const
{
    std::vector<std::reference_wrapper<const Layer>> layers;
    layers.reserve(skills.size());
    for (const auto &skill : skills) {
        layers.push_back(std::cref(skill.layer()));
    }
    return layers;
}

size_t Model::numLayers() const
{
    return skills.size();
}

Pattern Model::infer(const Pattern &input)
{
    if (skills.empty()) {
        throw std::runtime_error("No layers exist in the model to perform inference.");
    }

    Pattern output = input;
    for (const auto &skill : skills) {
        // cppcheck-suppress useStlAlgorithm
        output = skill.perform(output);
    }
    return output;
}
