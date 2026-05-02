#include "training/FeedforwardCoach.h"

#include "base/Model.h"
#include "training/Coach.h"

void FeedforwardCoach::learn(Model &network,
                               const Batch &inputs,
                               const Batch &labels,
                               Scalar learningRate,
                               size_t epochs)
{
    Coach coach;
    coach.practice(network, inputs, labels, learningRate, epochs);
}
