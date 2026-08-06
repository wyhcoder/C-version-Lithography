#include "fft.h"
#include <Eigen/Dense>
#include <iostream>



int main(){
    Eigen::MatrixXcd A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    std::cout << A << std::endl;
    litho::FFT::fftshift_inplace(A);
    std::cout << A << std::endl;
    litho::FFT::ifftshift_inplace(A);
    std::cout << A << std::endl;
    

    
}