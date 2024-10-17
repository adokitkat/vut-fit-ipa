/**
 * Three layers neural network
 * 
 * Source of sources: https://github.com/OmarAflak/Neural-Network
 * Author: Omar Aflak
 * Modified by: Tomas Goldmann, for education purposes
 * Date: 10/2019
 * Revision: 1
 */

#ifndef DEF_NETWORK
#define DEF_NETWORK

#include <vector>
#include <fstream>
#include <ostream>

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <iostream>

#include "../Matrix/Matrix.h"


//pro kazdou funkci definovanou v *.S je zapotřebí přidat zde deklaraci s extern
//extern "C" void f();


float sigmoid(float x);
float sigmoidePrime(float x);
float random(float x);


class Network
{
public:
    Network(std::vector<int> neurons, float learningRate, std::string neurons_file);
    Network(const char *filepath);

    Matrix computeOutput(std::vector<float> input);
    void learn(std::vector<float> expectedOutput);
    void simd_learn(std::vector<float> expectedOutput);


    void saveNetworkParams(const char *filepath);
    void loadNetworkParams(const char *filepath);

    std::vector<Matrix > W;
    std::vector<Matrix > B;
    
private:
    std::vector<Matrix > H;
    std::vector<Matrix > dEdW;
    std::vector<Matrix > dEdB;

    Matrix Y;

    int hiddenLayersCount;
    float learningRate;


    void printToFile(Matrix &m, std::ofstream &file);
};

#endif
