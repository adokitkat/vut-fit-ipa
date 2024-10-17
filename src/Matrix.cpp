#include "Matrix.h"
#include <intrin.h>

using namespace std;

#define ASM

Matrix::Matrix(int height, int width)
{
    this->height = height;
    this->width = width;

    this->array = new float*[height];
    for(int i = 0; i < height; ++i)
        this->array[i] = new float[width];
}


Matrix::Matrix(float ** data, int height, int width)
{
    assert((height*width)!=0);

    this->height = height;
    this->width = width;


    this->array = new float*[height];
    for(int i = 0; i < height; ++i)
        this->array[i] = new float[width];


    int i,j;
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            this->array[i][j] = data[i][j];
        }
    }
}

Matrix::Matrix(float * data, int length)
{
    assert((length)!=0);

    this->height = 1;
    this->width = length;


    this->array = new float*[this->height];
    for(int i = 0; i < this->height; ++i)
        this->array[i] = new float[this->width];


    int i,j;
    for (i=0 ; i<this->height ; i++)
    {
        for (j=0 ; j<this->width ; j++)
        {
            this->array[i][j] = data[j];
        }
    }
}


Matrix::Matrix(){}

int Matrix::getHeight()
{
    return height;
}

int Matrix::getWidth()
{
    return width;
}

void Matrix::fill(float const &value)
{
    int i,j;
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            array[i][j] = value;
        }
    }
}

void Matrix::put(int h, int w, float const &value)
{
    array[h][w] = value;
}

float Matrix::get(int h, int w) const
{
    return array[h][w];
}

Matrix Matrix::add(float const &value)
{
    int i, j;
#ifndef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            array[i][j] += value;
        }
    }
#else
    __m128 value1 = _mm_load1_ps((float*)&value);
    __m128 array4_0, array4_1, array4_2, array4_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);
            array4_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_3 = _mm_load_ps((float*)&array[i+3][j]);
            
            array4_0 = _mm_add_ps(array4_0, value1);
            array4_1 = _mm_add_ps(array4_1, value1);
            array4_2 = _mm_add_ps(array4_2, value1);
            array4_3 = _mm_add_ps(array4_3, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);
            _mm_store_ps((float*)&array[i+1][j], array4_1);
            _mm_store_ps((float*)&array[i+2][j], array4_2);
            _mm_store_ps((float*)&array[i+3][j], array4_3);
        }
        for (; j < width ; ++j) {
            array[i][j]   += value;
            array[i+1][j] += value;
            array[i+2][j] += value;
            array[i+3][j] += value;
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);

            array4_0 = _mm_add_ps(array4_0, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);
    
        }
        for (; j<width ; ++j) {
            array[i][j] += value;
        }
    }
#endif
    return *this;
}

Matrix Matrix::subtract(float const &value)
{
    int i, j;
#ifdef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            array[i][j] -= value;
        }
    }
#else
    __m128 value1 = _mm_load1_ps((float*)&value);
    __m128 array4_0, array4_1, array4_2, array4_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);
            array4_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_3 = _mm_load_ps((float*)&array[i+3][j]);
            
            array4_0 = _mm_sub_ps(array4_0, value1);
            array4_1 = _mm_sub_ps(array4_1, value1);
            array4_2 = _mm_sub_ps(array4_2, value1);
            array4_3 = _mm_sub_ps(array4_3, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);
            _mm_store_ps((float*)&array[i+1][j], array4_1);
            _mm_store_ps((float*)&array[i+2][j], array4_2);
            _mm_store_ps((float*)&array[i+3][j], array4_3);
        }
        for (; j < width ; ++j) {
            array[i][j]   -= value;
            array[i+1][j] -= value;
            array[i+2][j] -= value;
            array[i+3][j] -= value;
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);

            array4_0 = _mm_sub_ps(array4_0, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);
    
        }
        for (; j<width ; ++j) {
            array[i][j] -= value;
        }
    }
#endif
    return *this;
}

Matrix Matrix::multiply(float const &value)
{
    int i,j;
#ifndef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            array[i][j] *= value;
        }
    }
#else
    __m128 value1 = _mm_load1_ps((float*)&value);
    __m128 array4_0, array4_1, array4_2, array4_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);
            array4_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_3 = _mm_load_ps((float*)&array[i+3][j]);
            
            array4_0 = _mm_mul_ps(array4_0, value1);
            array4_1 = _mm_mul_ps(array4_1, value1);
            array4_2 = _mm_mul_ps(array4_2, value1);
            array4_3 = _mm_mul_ps(array4_3, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);
            _mm_store_ps((float*)&array[i+1][j], array4_1);
            _mm_store_ps((float*)&array[i+2][j], array4_2);
            _mm_store_ps((float*)&array[i+3][j], array4_3);
/*
            cout << endl << array[i][j]   << "\t" << array[i][j+1]   << "\t" << array[i][j+2]   << "\t" << array[i][j+3] << endl ;
            cout         << array[i+1][j] << "\t" << array[i+1][j+1] << "\t" << array[i+1][j+2] << "\t" << array[i+1][j+3] << endl ;
            cout         << array[i+2][j] << "\t" << array[i+2][j+1] << "\t" << array[i+2][j+2] << "\t" << array[i+2][j+3] << endl ;
            cout         << array[i+3][j] << "\t" << array[i+3][j+1] << "\t" << array[i+3][j+2] << "\t" << array[i+3][j+3] << endl ;
*/
        }
        for (; j < width ; ++j) {
            array[i][j]   *= value;
            array[i+1][j] *= value;
            array[i+2][j] *= value;
            array[i+3][j] *= value;
/*           
            cout << endl << array[i][j] << endl ;
            cout         << array[i+1][j] << endl;
            cout         << array[i+2][j] << endl;
            cout         << array[i+3][j] << endl;
*/
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_0 = _mm_load_ps((float*)&array[i][j]);

            array4_0 = _mm_mul_ps(array4_0, value1);

            _mm_store_ps((float*)&array[i][j], array4_0);

//            cout << endl << array[i][j] << "\t" << array[i][j+1] << "\t" << array[i][j+2] << "\t" << array[i][j+3] << endl ;
    
        }
        for (; j<width ; ++j) {
            array[i][j] *= value;
//            cout << endl << array[i][j] << endl ;
        }
    }
#endif
    return *this;
}

Matrix Matrix::add(Matrix const &m) const
{
    assert(height==m.height && width==m.width);

    Matrix result(height, width);
    //f_matrix(result.array, height, width);
    int i,j;
#ifndef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }
#else
    __m128 array4_1_0, array4_2_0, array4_1_1, array4_2_1, array4_1_2, array4_2_2, array4_1_3, array4_2_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {

            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2_1 = _mm_load_ps((float*)&m.array[i+1][j]);

            array4_1_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_2_2 = _mm_load_ps((float*)&m.array[i+2][j]);

            array4_1_3 = _mm_load_ps((float*)&array[i+3][j]);
            array4_2_3 = _mm_load_ps((float*)&m.array[i+3][j]);

            array4_1_0 = _mm_add_ps(array4_1_0, array4_2_0);
            array4_1_1 = _mm_add_ps(array4_1_1, array4_2_1);
            array4_1_2 = _mm_add_ps(array4_1_2, array4_2_2);
            array4_1_3 = _mm_add_ps(array4_1_3, array4_2_3);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
            _mm_store_ps((float*)&result.array[i+1][j], array4_1_1);
            _mm_store_ps((float*)&result.array[i+2][j], array4_1_2);
            _mm_store_ps((float*)&result.array[i+3][j], array4_1_3);
        }
        for (; j < width ; ++j) {
            result.array[i][j]   = array[i][j]   + m.array[i][j];
            result.array[i+1][j] = array[i+1][j] + m.array[i+1][j];
            result.array[i+2][j] = array[i+2][j] + m.array[i+2][j];
            result.array[i+3][j] = array[i+3][j] + m.array[i+3][j];
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_0 = _mm_add_ps(array4_1_0, array4_2_0);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
    
        }
        for (; j<width ; ++j) {
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }
#endif
    return result;
}

Matrix Matrix::subtract(Matrix const &m) const
{
    assert(height==m.height && width==m.width);

    Matrix result(height, width);
    int i,j;
#ifndef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }
#else
    __m128 array4_1_0, array4_2_0, array4_1_1, array4_2_1, array4_1_2, array4_2_2, array4_1_3, array4_2_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {

            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2_1 = _mm_load_ps((float*)&m.array[i+1][j]);

            array4_1_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_2_2 = _mm_load_ps((float*)&m.array[i+2][j]);

            array4_1_3 = _mm_load_ps((float*)&array[i+3][j]);
            array4_2_3 = _mm_load_ps((float*)&m.array[i+3][j]);

            array4_1_0 = _mm_sub_ps(array4_1_0, array4_2_0);
            array4_1_1 = _mm_sub_ps(array4_1_1, array4_2_1);
            array4_1_2 = _mm_sub_ps(array4_1_2, array4_2_2);
            array4_1_3 = _mm_sub_ps(array4_1_3, array4_2_3);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
            _mm_store_ps((float*)&result.array[i+1][j], array4_1_1);
            _mm_store_ps((float*)&result.array[i+2][j], array4_1_2);
            _mm_store_ps((float*)&result.array[i+3][j], array4_1_3);
        }
        for (; j < width ; ++j) {
            result.array[i][j]   = array[i][j]   - m.array[i][j];
            result.array[i+1][j] = array[i+1][j] - m.array[i+1][j];
            result.array[i+2][j] = array[i+2][j] - m.array[i+2][j];
            result.array[i+3][j] = array[i+3][j] - m.array[i+3][j];
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_0 = _mm_sub_ps(array4_1_0, array4_2_0);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
    
        }
        for (; j<width ; ++j) {
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }
#endif
    return result;
}

Matrix Matrix::multiply(Matrix const &m) const
{
    assert(height==m.height && width==m.width);

    Matrix result(height, width);
    int i,j;
#ifndef ASM
    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }
#else
    __m128 array4_1_0, array4_2_0, array4_1_1, array4_2_1, array4_1_2, array4_2_2, array4_1_3, array4_2_3;
    i = 0;
    for (; i + 3 < height; i+=4) {
        j = 0;
        for (; j + 3 < width; j+=4) {

            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_1 = _mm_load_ps((float*)&array[i+1][j]);
            array4_2_1 = _mm_load_ps((float*)&m.array[i+1][j]);

            array4_1_2 = _mm_load_ps((float*)&array[i+2][j]);
            array4_2_2 = _mm_load_ps((float*)&m.array[i+2][j]);

            array4_1_3 = _mm_load_ps((float*)&array[i+3][j]);
            array4_2_3 = _mm_load_ps((float*)&m.array[i+3][j]);

            array4_1_0 = _mm_mul_ps(array4_1_0, array4_2_0);
            array4_1_1 = _mm_mul_ps(array4_1_1, array4_2_1);
            array4_1_2 = _mm_mul_ps(array4_1_2, array4_2_2);
            array4_1_3 = _mm_mul_ps(array4_1_3, array4_2_3);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
            _mm_store_ps((float*)&result.array[i+1][j], array4_1_1);
            _mm_store_ps((float*)&result.array[i+2][j], array4_1_2);
            _mm_store_ps((float*)&result.array[i+3][j], array4_1_3);
        }
        for (; j < width ; ++j) {
            result.array[i][j]   = array[i][j]   * m.array[i][j];
            result.array[i+1][j] = array[i+1][j] * m.array[i+1][j];
            result.array[i+2][j] = array[i+2][j] * m.array[i+2][j];
            result.array[i+3][j] = array[i+3][j] * m.array[i+3][j];
        }
    }
    for (; i < height; ++i) {
        j = 0;
        for (; j + 3 < width; j+=4) {
            
            array4_1_0 = _mm_load_ps((float*)&array[i][j]);
            array4_2_0 = _mm_load_ps((float*)&m.array[i][j]);

            array4_1_0 = _mm_mul_ps(array4_1_0, array4_2_0);

            _mm_store_ps((float*)&result.array[i][j], array4_1_0);
    
        }
        for (; j<width ; ++j) {
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }
#endif
    return result;
}

Matrix Matrix::dot(Matrix const &m) const
{

    assert(width==m.height);

    int i,j,h, mwidth = m.width;
    float w=0;

    Matrix result(height, mwidth);

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<mwidth ; j++)
        {
            for (h=0 ; h<width ; h++)
            {
                w += array[i][h]*m.array[h][j];
            }
            result.array[i][j] = w;
            w=0;
        }
    }

    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(width, height);
    int i,j;

#ifdef ASM

    for (i=0 ; i<width ; i++){
        for (j=0 ; j<height ; j++){
            result.array[i][j] = array[j][i];
        }
    }

#else

    __m128 array4_0, array4_1, array4_2, array4_3;
    i = 0, j = 0;
    int g = 0;
    for (; i + 3 < height; i+=4) {
        
        
        for (; j + 3 < width; j+=4) {

            
            array4_0 = _mm_load_ps((float*)&array[i*j][j]);
            array4_1 = _mm_load_ps((float*)&array[i*j+1][j]);
            array4_2 = _mm_load_ps((float*)&array[i*j+2][j]);
            array4_3 = _mm_load_ps((float*)&array[i*j+3][j]);
            
            _MM_TRANSPOSE4_PS(array4_0, array4_1, array4_2, array4_3);

            _mm_store_ps((float*)&result.array[i*j][j], array4_0);
            _mm_store_ps((float*)&result.array[i*j+1][j], array4_1);
            _mm_store_ps((float*)&result.array[i*j+2][j], array4_2);
            _mm_store_ps((float*)&result.array[i*j+3][j], array4_3);

        }
        
        for (; j < width; ++j) {
            result.array[j][i]   = array[i][j];
            result.array[j][i+1] = array[i+1][j];
            result.array[j][i+2] = array[i+2][j];
            result.array[j][i+3] = array[i+3][j];
        }
    }
    for (i; i < height; ++i) {
        j = 0;
        for (; j < width ; ++j) {
            result.array[j][i] = array[i][j];
        }
    }
    /*
    for (i; i < width; ++i) {
        j = 0;
        for (; j < height ; ++j) {
            result.array[i][j] = array[j][i];
        }
    }
    */
#endif

    return result;
}


Matrix Matrix::applyFunction(float (*function)(float)) const
{
    Matrix result(height, width);
    int i,j;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++){
            float in=this->array[i][j];
            float x= (*function)(in);
            result.array[i][j] = x;
        }
    }

    return result;
}

Matrix Matrix::subMatrix(int startH, int startW, int h, int w) const
{
    assert(startH+h<=height && startW+w<=width);

    Matrix result(h,w);
    int i,j;

    for (i=startH ; i<startH+h ; i++)
    {
        for (j=startW ; j<startW+w ; j++)
        {
            result.array[i-startH][j-startW] = array[i][j];
        }
    }
    return result;
}

void Matrix::print(std::ostream &flux) const
{
    int i,j;
    int maxLength[width] = {};
    std::stringstream ss;

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            ss << array[i][j];
            if(maxLength[j] < ss.str().size())
            {
                maxLength[j] = ss.str().size();
            }
            ss.str(std::string());
        }
    }

    for (i=0 ; i<height ; i++)
    {
        for (j=0 ; j<width ; j++)
        {
            flux << array[i][j];
            ss << array[i][j];
            for (int k=0 ; k<maxLength[j]-ss.str().size()+1 ; k++)
            {
                flux << " ";
            }
            ss.str(std::string());
        }
        flux << std::endl;
    }
}

bool Matrix::operator==(Matrix const &m)
{
    if(height==m.height && width==m.width)
    {
        int i,j;
        for (i=0 ; i<height ; i++)
        {
            for (j=0 ; j<width ; j++)
            {
                if(array[i][j]!=m.array[i][j])
                {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

bool Matrix::operator!=(Matrix const &m)
{
    return !operator==(m);
}

void Matrix::operator+=(Matrix const &m)
{
    this->array = add(m).array;
}

void Matrix::operator-=(Matrix const &m)
{
    this->array = subtract(m).array;
}

void Matrix::operator*=(Matrix const &m)
{
    this->array = multiply(m).array;
}

void Matrix::operator+=(float const &m)
{
    add(m);
}

void Matrix::operator-=(float const &m)
{
    subtract(m);
}

void Matrix::operator*=(float const &m)
{
    multiply(m);
}

float& Matrix::operator()(int y, int x)
{
    assert(y<height && x<width);
    return array[y][x];
}

Matrix operator+(Matrix const &a, Matrix const &b)
{
    return a.add(b);
}

Matrix operator-(Matrix const &a, Matrix const &b)
{
    return a.subtract(b);
}

Matrix operator*(Matrix const &a, Matrix const &b)
{
    return a.multiply(b);
}

Matrix operator+(Matrix const &a, float const &b)
{
    Matrix result = a;
    return result.add(b);
}

Matrix operator-(Matrix const &a, float const &b)
{
    Matrix result = a;
    return result.subtract(b);
}

Matrix operator*(Matrix const &a, float const &b)
{
    Matrix result = a;
    return result.multiply(b);
}

std::ostream& operator<<(std::ostream &flux, Matrix const &m)
{
    m.print(flux);
    return flux;
}