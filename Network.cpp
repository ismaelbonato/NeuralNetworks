#include "Network.h"
#include "Neuron.h"

int Network::threshld(double k)
{
    return ((k >= 0) ? 1 : -1);
}

void Network::activation(const Pattern &pattern)
{   
    output.clear();
    for (auto &neuron : neurons) {
        for(int j=0;j< neuron.weights.size();j++)
    {
        //std::cout <<"\n nrn["<<"].weightv["<<j<<"] is " <<neuron.weights[j];
    }

        neuron.activation = neuron.act(pattern);
        //std::cout << "Neuron activation: " << neuron.activation << std::endl;
        output.emplace_back(threshld(neuron.activation));
    }
}

RowOfWeights &Network::hebbianLearning(const std::vector<Pattern> &patterns)
{
    size_t pSize = patterns.size();
    
    // Initialize rowOfWeights with correct dimensions
    rowOfWeights.resize(patterns.at(0).size(), std::vector<double>(patterns.at(0).size(),0.0));

    std::cout << "Size of the pattern: " <<  patterns.at(0).size() << std::endl;

    for (size_t i = 0; i < patterns.at(0).size(); ++i) {
        for (size_t j = 0; j < patterns.at(0).size(); ++j) {
            if (i == j) {
                rowOfWeights[i][j] = 0; // No self-connection
                continue;
            }
            double sum = 0;
            for (const auto &pattern : patterns) {
                sum += bin_to_bipolar(pattern[i]) * bin_to_bipolar(pattern[j]);
            }
            rowOfWeights[i][j] = sum / static_cast<double>(pSize);
        }
    }
    return rowOfWeights;
}

void Network::updateNeurons(const RowOfWeights &rw)
{
    // Update neurons with new weights
    neurons.clear();
    neurons.reserve(rowOfWeights.at(0).size());

    for (auto &ws : rw) {
        neurons.emplace_back(ws);
    }

    std::cout << "amount of neurons: " << neurons.size() << std::endl;
}

void Network::printOutput()
{
    std::cout << "print Output " << std::endl;
    printPattern(output);
}

void Network::printPattern(const Pattern &p)
{
    std::cout << "print Pattern " << std::endl;
    size_t idx = 0;
    for (auto i : p) {
        if (idx >= p.size()/64) {
            std::cout << std::endl;
            idx = 0;
        }
        idx++;

   /*    if (i < -1)
            std::cout << i << ",";
        else 
            std::cout << " " << i << ",";
        */
        std::cout << (i > 0 ? "x": " ");
        //std::cout << i << "; ";
    }
    std::cout << std::endl;

}

void Network::printWeights()
{
    std::cout << "printWeights " << std::endl;
    size_t idx = 0;
    for (auto &ws : rowOfWeights) {
        auto wSize = ws.size();
        for (auto w : ws) {
            if (idx >= wSize) {
                std::cout << std::endl;
                idx = 0;
            }
            idx++;
            
            if (w < -1)
                std::cout << w << ",";
            else 
                std::cout << " " << w << ",";
        }
    }
    std::cout << std::endl;

}