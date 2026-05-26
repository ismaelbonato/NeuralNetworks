#pragma once

#include "base/Model.h"

#include <string>

namespace nn::serialization {

void saveModelToFile(const Model &model, const std::string &path);

} // namespace nn::serialization
