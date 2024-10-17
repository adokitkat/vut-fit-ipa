 /**
 * Three layers neural network
 * 
 * Source of sources: https://github.com/OmarAflak/Neural-Network
 * Author: Omar Aflak
 * Modified by: Tomas Goldmann for education purposes
 * Date: 10/2019
 * Revision: 1
 */


#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>

#include "Network/Network.h"


#define ITERATION_COUNT 30
using namespace std;

float stepFunction(float x);
void loadTraining(const char *filename, vector<vector<float> > &input, vector<vector<float> > &output);


