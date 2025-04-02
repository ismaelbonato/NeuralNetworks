#include "Network.h"
#include "Neuron.h"

int Network::threshld(int k)
{
    if (k >= 0)
        return (1);
    else
        return (0);
}

Network::Network(Weights &a, Weights &b, Weights &c, Weights &d)
{
    neurons.emplace_back(a);
    neurons.emplace_back(b);
    neurons.emplace_back(c);
    neurons.emplace_back(d);
}

void Network::activation(Pattern &patrn)
{
    output.clear();
    
    for(auto &neuron : neurons) {
        for(auto weight : neuron.weights) {
            std::cout << "\n Neurons[] is " << weight;
        }

        neuron.activation = neuron.act(4,patrn);
        
        std::cout << "\nactivation is " << neuron.activation;
        auto out = output.emplace_back(threshld(neuron.activation));
                
        std::cout<< "\noutput value is " << out << "\n";
    }
}