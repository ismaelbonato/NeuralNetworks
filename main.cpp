#include <iostream>
#include <math.h>
#include <array>
#include <memory>
#include <filesystem>
#include "Helper.h"


#include "Hopfield.h"
#include "LearningRule.h"
#include "Layer.h"

#include "Perceptron.h"


int main()
{
/*  std::vector<Pattern> patterns;

    patterns.emplace_back(png_to_bits("../Misc/bart.png"));
    patterns.emplace_back(png_to_bits("../Misc/homer.png"));
    patterns.emplace_back(png_to_bits("../Misc/marge.png"));
    patterns.emplace_back(png_to_bits("../Misc/meg.png"));
    patterns.emplace_back(png_to_bits("../Misc/grandpa.png"));
    //patterns.emplace_back(png_to_bits("../Misc/lisa.png"));
    Pattern p(png_to_bits("../Misc/homer_defect.png"));

    auto N = patterns.at(0).size();

    Hopfield net(std::move(std::make_unique<HebbianRule>()), N);

    net.learn(patterns);
    
    auto ret = net.forward(p);

    print(p);
    print(ret);
    std::cout << "Hello, Hopfield Network!" << std::endl;
*/

    std::vector<Pattern> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<Pattern> labels = {
        {0.0}, // 0 AND 0
        {0.0}, // 0 AND 1
        {0.0}, // 1 AND 0
        {1.0}  // 1 AND 1
    };

    size_t in = inputs.at(0).size();
    size_t out = labels.at(0).size();
    Perceptron net(std::move(std::make_unique<PerceptronRule>()), in, out);

    net.learn(inputs, labels);

    std::cout << "Perceptron Network trained!" << std::endl;
    for (const auto& input : inputs) {
        Pattern output = net.forward(input);
        print(input); 
        print(output); 
    }

    return 0;
}