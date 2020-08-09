# FFT-MPI-OpenMP-CUDA
 Parallel FFT for big integer multiplication. Written in three versions: MPI, OpenMP and CUDA(cufft). It is a course assignment of *MPI program design* given by *Prof. Xiang*, *2019 Fall*, *SMS*, *Nankai Univ*.  
 大整数乘法的并行FFT，MPI程序设计课程作业，包括MPI、OpenMP、cuFFT三个版本。
 
 For more details, please see the **[report](https://github.com/Alexhaoge/FFT-MPI-OpenMP-CUDA/blob/master/Report.pdf)**.

## OpenMP version
parallel the doacross loop in serial version.

## CUDA version
use cuFFT.

## MPI version
For larger data (FFT bits >= 10^8) which cannot be stored in a single processor (and GPU), MPI with distributed memory is a better solution.

Say the total bits is *2^n* and we have *2^p* processors.  
In the first *p* round **butterfly operation**, all the processors can work independently, and then each processor needs to share data with a particular processor since *p+1* round.  

In *MPI_FFT.cpp*, **MPI_Sendrecv** is used for data sharing of a pair of processors and **MPI_Alltoall** is used to swap the positions of all processors bitwisely.

## Precision Issue
A flaw of these program is the precision of complex numbers in FFT cannot be guaranteed when *2^n* is very large. In big integer multiplication where the final result will be rounded integers, the precision does not matter. But for other application of FFT, that may be a problem.
