#include "training/Coach.h"

#include <memory>
#include <stdexcept>

Coach::Coach()
    : Coach(std::make_unique<BackpropagationPracticePlan>())
{}

Coach::Coach(std::unique_ptr<PracticePlan> newPracticePlan)
    : practicePlan(std::move(newPracticePlan))
{
    if (!practicePlan) {
        throw std::invalid_argument("Coach requires a practice plan.");
    }
}

Coach::~Coach() = default;

void Coach::practice(Model &network,
                     const PracticeData &data,
                     const PracticeOptions &options) const
{
    practicePlan->practice(network, data, options);
}
