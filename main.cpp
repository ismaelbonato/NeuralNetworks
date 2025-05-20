#include "Network.h"
#include "Neuron.h"
#include <iostream>
#include <math.h>
#include <array>
#include <memory>
#include <filesystem>

#include <opencv2/opencv.hpp>

Pattern png_to_bits(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE); 
    cv::Mat resizedImage;

    cv::resize(img, resizedImage, cv::Size(64, 64));

    Pattern pattern;
    pattern.reserve(static_cast<size_t>(resizedImage.rows) * static_cast<size_t>(resizedImage.cols));
    
    for (size_t r = 0; r < static_cast<size_t>(resizedImage.rows); ++r) {
        for (size_t c = 0; c < static_cast<size_t>(resizedImage.cols); ++c) {
            auto pixel = resizedImage.at<uchar>( static_cast<int>(r), static_cast<int>(c)); 
            pattern.emplace_back( pixel > 128 ? 1 : 0);
        }
    }
    return pattern;
}

// Compute Hamming distance between two patterns
int hamming_distance(const Pattern& a, const Pattern& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++dist;
    }
    return dist;
}


int main (void)
{
    std::vector<Pattern> patterns;

    patterns.emplace_back(png_to_bits("../Misc/bart.png"));
    patterns.emplace_back(png_to_bits("../Misc/homer.png"));
    patterns.emplace_back(png_to_bits("../Misc/marge.png"));
    patterns.emplace_back(png_to_bits("../Misc/meg.png"));
    patterns.emplace_back(png_to_bits("../Misc/grandpa.png"));
    Pattern p(png_to_bits("../Misc/homer_defect.png"));


/*  
    std::string dir = "../Misc/fonts/";
    int idx = 0;
   for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".png") {

            std::cout << entry.path().string() << std::endl;
            patterns.emplace_back(png_to_bits(entry.path().string()));
            ++idx;
        }
        if (idx == 5) break;
    }
  
   
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            std::cout << "Hamming(" << i << "," << j << ") = "
                      << hamming_distance(patterns[i], patterns[j]) << std::endl;
        }
    }
    */

    Network hopfieldNetwork;

    std::cout << "Learning weights using Hebbian rule" << std::endl;
    auto learnedWeights = hopfieldNetwork.hebbianLearning(patterns);

    //hopfieldNetwork.printWeights();
    
//    Pattern test{patterns.at(1)};

    for (auto &i : p) {
        i = bin_to_bipolar(i);
    }

    std::cout << "Initializing network with learned weights" << std::endl;
    hopfieldNetwork.updateNeurons(learnedWeights);
    std::cout << "Activating the network" << std::endl;
    hopfieldNetwork.printPattern(p);
    hopfieldNetwork.activation(p);

    hopfieldNetwork.printOutput();

    return 0;
}