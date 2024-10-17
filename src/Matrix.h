/**
 * Matrix library
 * 
 * Source of sources: https://github.com/OmarAflak/Neural-Network
 * Author: Omar Aflak
 * Modified by: Tomas Goldmann, for education purposes
 * Date: 10/2019
 * Revision: 1
 */

#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <assert.h>

//pro kazdou funkci definovanou v *.S je zapotřebí přidat zde deklaraci s extern
extern "C" void f_matrix(float ** data, int height, int width);


class Matrix
{
public:
    Matrix(int height, int width);
    Matrix(float ** arrray, int width, int height);
    Matrix(float * arrray, int width);
    Matrix();

    int getHeight();
    int getWidth();

    void fill(float const &value);
    void put(int h, int w, float const &value);
    float get(int h, int w) const;

    Matrix add(float const &value);
    Matrix subtract(float const &value);
    Matrix multiply(float const &value);

    Matrix add(Matrix const &m) const;
    Matrix subtract(Matrix const &m) const;
    Matrix multiply(Matrix const &m) const;
    Matrix dot(Matrix const &m) const;
    Matrix transpose() const;

    Matrix applyFunction(float (*function)(float)) const;
    Matrix subMatrix(int startH, int startW, int h, int w) const;
    void print(std::ostream &flux) const;

    bool operator==(Matrix const &m);
    bool operator!=(Matrix const &m);
    void operator+=(Matrix const &m);
    void operator-=(Matrix const &m);
    void operator*=(Matrix const &m);
    void operator+=(float const &m);
    void operator-=(float const &m);
    void operator*=(float const &m);
    float& operator()(int y, int x);

private:
    float ** array;
    int height;
    int width;
};

Matrix operator+(Matrix const &a, Matrix const &b);

Matrix operator-(Matrix const &a, Matrix const &b);

Matrix operator*(Matrix const &a, Matrix const &b);

Matrix operator+(Matrix const &a, float const &b);

Matrix operator-(Matrix const &a, float const &b);

Matrix operator*(Matrix const &a, float const &b);



std::ostream& operator<<(std::ostream &flux, Matrix const &m);

#endif


