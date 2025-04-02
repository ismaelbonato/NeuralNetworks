#include "Network.h"
#include "Neuron.h"
#include <iostream>
#include <math.h>
#include <array>

int main (void)
{
    /* 
    Weights wt1 = {0,-3,3,-3};
    Weights wt2 = {-3,0,-3,3};
    Weights wt3 = {3,-3,0,-3};
    Weights wt4 = {-3,3,-3,0};
    */
    Weights wt1 = { 0, -5,  4,  4};
    Weights wt2 = {-5,  0,  4,  4};
    Weights wt3 = { 4,  4,  0, -5};
    Weights wt4 = { 4,  4, -5,  0};

    Network network(wt1,wt2,wt3,wt4);
    Pattern patrn1 = {1,0,1,0};
    network.activation(patrn1);

    for(int idx = 0; idx < 4; ++idx)
    {
        if (network.output.at(idx) == patrn1.at(idx)) { 
            std::cout << "\n pattern= " << patrn1.at(idx) << " output = " <<network.output.at(idx)<<" component matches";
        } else { 
            std::cout << "\n pattern= " << patrn1.at(idx) << " output = " <<network.output.at(idx)<< " discrepancy occurred";
        }
    }
    
    std::cout << std::endl;
    Pattern patrn2 = {0,1,0,1};
    network.activation(patrn2);
    
    for(int idx = 0; idx < 4; ++idx)
    {
        if (network.output.at(idx) == patrn2.at(idx)) {
            std::cout << "\n pattern= " << patrn2.at(idx)  << " output = " << network.output.at(idx) << " component matches";
        } else {
            std::cout << "\n pattern= " << patrn2.at(idx)  << " output = " << network.output.at(idx) << " discrepancy occurred";
        }
    }

    std::cout << std::endl;
    Pattern patrn3 = {1, 0, 0, 1};
    network.activation(patrn3);
    
    for(int idx = 0; idx < 4; ++idx)
    {
        if (network.output.at(idx) == patrn3.at(idx)) {
            std::cout << "\n pattern= " << patrn3.at(idx)  << " output = " << network.output.at(idx) << " component matches";
        } else {
            std::cout << "\n pattern= " << patrn3.at(idx)  << " output = " << network.output.at(idx) << " discrepancy occurred";
        }
    }

    std::cout << std::endl;
    Pattern patrn4 = {0, 1, 1, 0};
    network.activation(patrn4);
    
    for(int idx = 0; idx < 4; ++idx)
    {
        if (network.output.at(idx) == patrn4.at(idx)) {
            std::cout << "\n pattern= " << patrn4.at(idx)  << " output = " << network.output.at(idx) << " component matches";
        } else {
            std::cout << "\n pattern= " << patrn4.at(idx)  << " output = " << network.output.at(idx) << " discrepancy occurred";
        }
    }



    return 0;
}