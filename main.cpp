#include "Helper.h"
#include <iostream>



int main()
{
    std::cout << "Hello, Neural Networks!" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Perceptron Network" << std::endl;
    std::cout << "==========================" << std::endl;
    perceptronNetwork();
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Feed Forward Network" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << std::endl;

    perceptronNaturalSelection();
    
    return 0;
}
