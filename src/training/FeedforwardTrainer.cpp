#include "training/FeedforwardTrainer.h"

#include "base/Model.h"
#include "training/Coach.h"

void FeedforwardTrainer::learn(Model &network,
                               const Batch &inputs,
                               const Batch &labels,
                               Scalar learningRate,
                               size_t epochs)
{
    Coach coach;
    coach.practice(network, inputs, labels, learningRate, epochs);
}
