#include "Network.h"

//********************************************************************Functions for optimalization ***************************************************/
Matrix Network::computeOutput(std::vector<float> input)
{
    H[0] = Matrix({input.data()},(int) input.size()); // row matrix

    for (int i=1 ; i<hiddenLayersCount+2 ; i++)
    {
        H[i] = H[i-1].dot(W[i-1]).add(B[i-1]).applyFunction(sigmoid);
    }

    return H[hiddenLayersCount+1];
}

void Network::learn(std::vector<float> expectedOutput)
{
    Y = Matrix(expectedOutput.data(), (int) expectedOutput.size()); // row matrix

    // Error E = 1/2 (expectedOutput - computedOutput)^2
    // Then, we need to calculate the partial derivative of E with respect to W and B
    // compute gradients
    dEdB[hiddenLayersCount] = H[hiddenLayersCount+1].subtract(Y).multiply(H[hiddenLayersCount].dot(W[hiddenLayersCount]).add(B[hiddenLayersCount]).applyFunction(sigmoidePrime));
    
    //dEdB[hiddenLayersCount].print(std::cout);
    //std::cout<< "----" << std::endl;
    for (int i=hiddenLayersCount-1 ; i>=0 ; i--)
    {
        dEdB[i] = dEdB[i+1].dot(W[i+1].transpose()).multiply(H[i].dot(W[i]).add(B[i]).applyFunction(sigmoidePrime));
    }

    for (int i=0 ; i<hiddenLayersCount+1 ; i++)
    {
        dEdW[i] = H[i].transpose().dot(dEdB[i]);
    }

    // update weights
    for (int i=0 ; i<hiddenLayersCount+1 ; i++)
    {
        W[i] = W[i].subtract(dEdW[i].multiply(learningRate));
        B[i] = B[i].subtract(dEdB[i].multiply(learningRate));
    }
}


void Network::simd_learn(std::vector<float> expectedOutput)
{
    Y = Matrix(expectedOutput.data(), (int) expectedOutput.size()); // row matrix

    // Error E = 1/2 (expectedOutput - computedOutput)^2
    // Then, we need to calculate the partial derivative of E with respect to W and B
    // compute gradients
    dEdB[hiddenLayersCount] = H[hiddenLayersCount+1].subtract(Y).multiply(H[hiddenLayersCount].dot(W[hiddenLayersCount]).add(B[hiddenLayersCount]).applyFunction(sigmoidePrime));
    
    //dEdB[hiddenLayersCount].print(std::cout);
    //std::cout<< "----" << std::endl;
    for (int i=hiddenLayersCount-1 ; i>=0 ; i--)
    {
        dEdB[i] = dEdB[i+1].dot(W[i+1].transpose()).multiply(H[i].dot(W[i]).add(B[i]).applyFunction(sigmoidePrime));
    }

    for (int i=0 ; i<hiddenLayersCount+1 ; i++)
    {
        dEdW[i] = H[i].transpose().dot(dEdB[i]);
    }

    // update weights
    for (int i=0 ; i<hiddenLayersCount+1 ; i++)
    {
        W[i] = W[i].subtract(dEdW[i].multiply(learningRate));
        B[i] = B[i].subtract(dEdB[i].multiply(learningRate));
    }
}

/*******************************************************************************************************************************************************/


Network::Network(std::vector<int> neurons, float learningRate, std::string neurons_file )
{
    srand (time(NULL));

    this->learningRate = learningRate;
    this->hiddenLayersCount = neurons.size()-2;

    H = std::vector<Matrix >(hiddenLayersCount+2);
    W = std::vector<Matrix >(hiddenLayersCount+1);
    B = std::vector<Matrix >(hiddenLayersCount+1);
    dEdW = std::vector<Matrix >(hiddenLayersCount+1);
    dEdB = std::vector<Matrix >(hiddenLayersCount+1);

    std::fstream file_B;
    std::fstream file_W;
    std::ifstream infile("B"+neurons_file);
    bool read=false;
    if (infile.good())
    {
        infile.close();

        file_B.open("B"+neurons_file,std::ios::in);
        file_W.open("W"+neurons_file,std::ios::in);
        read=true;
    }
    else
    {
        file_B.open("B"+neurons_file,std::ios::out);
        file_W.open("W"+neurons_file,std::ios::out);
    }
    
    for (int i=0 ; i<neurons.size()-1 ; i++)
    {
        W[i] = Matrix(neurons[i], neurons[i+1]);
        B[i] = Matrix(1, neurons[i+1]);

        W[i] = W[i].applyFunction(random);
        B[i] = B[i].applyFunction(random);

        if(read)
        {
            for (int ix=0;ix<W[i].getHeight();ix++)
            {
                std::string line;
                std::getline(file_W,line);
                std::vector<float> line_float_values;

                std::istringstream iss(line);
                for (float d; iss >> d; ) { line_float_values.push_back(d); }


                for (int j=0;j<W[i].getWidth();j++)
                {
                    W[i].put(ix,j,line_float_values.at(j));
                }
            }

            for (int ix=0;ix<B[i].getHeight();ix++)
            {
                std::string line;
                std::getline(file_B,line);
                std::vector<float> line_float_values;

                std::istringstream iss(line);
                for (float d; iss >> d; ) { line_float_values.push_back(d); }

                for (int j=0;j<B[i].getWidth();j++)
                {
                    B[i].put(ix,j,line_float_values.at(j));
                }
            }
        }
        else
        {
            for (int ix=0;ix<W[i].getHeight();ix++)
            {
                std::string line;

                for (int j=0;j<W[i].getWidth();j++)
                {
                    line=line + " " + std::to_string(W[i].get(ix,j));
                }
                file_W << line << std::endl;
            }

            for (int ix=0;ix<B[i].getHeight();ix++)
            {
                std::string line;

                for (int j=0;j<B[i].getWidth();j++)
                {
                    line=line + " " + std::to_string(B[i].get(ix,j));
                }
                file_B << line << std::endl;
            }
        }
        
    }
    file_W.close();
    file_B.close();

}

Network::Network(const char *filepath)
{
    loadNetworkParams(filepath);
}



void Network::printToFile(Matrix &m, std::ofstream &file)
{
    int h = m.getHeight();
    int w = m.getWidth();

    file << h << std::endl;
    file << w << std::endl;
    for (int i=0 ; i<h ; i++)
    {
        for (int j=0 ; j<w ; j++)
        {
            file << m.get(i,j) << (j!=w-1?" ":"");
        }
        file << std::endl;
    }
}

void Network::saveNetworkParams(const char *filepath)
{
    std::ofstream out(filepath);

    out << hiddenLayersCount << std::endl;
    out << learningRate << std::endl;

    for (Matrix m : W){
        printToFile(m, out);
    }

    for (Matrix m : B){
        printToFile(m, out);
    }

    out.close();
}

void Network::loadNetworkParams(const char *filepath)
{
    std::ifstream in(filepath);
    std::vector<Matrix > params;
    float val;
    int h,w;

    if(in)
    {
        in >> hiddenLayersCount;
        in >> learningRate;

        H = std::vector<Matrix >(hiddenLayersCount+2);
        W = std::vector<Matrix >(hiddenLayersCount+1);
        B = std::vector<Matrix >(hiddenLayersCount+1);
        dEdW = std::vector<Matrix >(hiddenLayersCount+1);
        dEdB = std::vector<Matrix >(hiddenLayersCount+1);

        for(int i=0 ; i<2*hiddenLayersCount+2 ; i++)
        {
            in >> h;
            in >> w;
            Matrix m(h,w);
            for (int hh=0 ; hh<h ; hh++)
            {
                for (int ww=0 ; ww<w ; ww++)
                {
                    in >> val;
                    m.put(hh,ww,val);
                }
            }

            params.push_back(m);
        }
    }
    in.close();

    // assign values
    for (int i=0 ; i<hiddenLayersCount+1 ; i++)
    {
        W[i] = params[i];
    }

    for (int i=hiddenLayersCount+1 ; i<params.size() ; i++)
    {
        B[i-hiddenLayersCount-1] = params[i];
    }
}

float random(float x)
{
    float result=(float)(rand() % 10000 + 1)/10000-0.5;
    return result;
}

float sigmoid(float x)
{
    double e=std::exp( -x);
    float result=1.0/(float)(1+e);
    return result;
}

float sigmoidePrime(float x)
{
    float result=std::exp(-x)/ pow(1+std::exp(-x), 2);
    return result;
}


