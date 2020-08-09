#!/bin/bash
# shell script to automate program tests
date >> log/sfft.log
date >> log/cufft.log
Thread=1
num=0
g++ -lm -O3 -o sfft sfft1.cpp -w
g++ -fopenmp -lm -O3 -o fft2 fft2.cpp -w
g++ -fopenmp -lm -O3 -o fft1 fft1.cpp -w
nvcc cufft.cu -lcufft -O3 -o cufft -w
mpicxx MPIFFT.cpp -lm -o mpifft -O3 -w

echo compile_done
for j in 9
do
  mkdir log/d$j
  date > log/d$j/mpi.log
  echo $j hahahah
  echo -e "d$j\t\c" >> log/sfft.log
  for i in 0
  do
    cp data/d$j/fft$i.in fft.in
    echo case$i input_done

    nohup timeout 300 ./sfft >> log/sfft.log &
    echo case$i sfft_done
    echo case$i cufft_done 
    proc=3
    while(($proc<=7))
    do
      let "y=2**$proc"
      echo mpi $y
      mpiexec -n $y ./mpifft>> log/d$j/mpi.log
      echo -e "\t\c">> log/d$j/mpi.log
      proc=$((proc+1))
    done
    echo case$i mpifft_done
    echo -e "">> log/d$j/mpi.log
  done

  echo -e "" >> log/sfft.log
  echo -e "" >> log/cufft.log
done

for j in 15 16 17 18 19 20 21 22 23 24 25
do
  mkdir log/d$j
  date > log/d$j/mpi.log
  echo $j hahahah
  echo -e "d$j\t\c" >> log/sfft.log
  echo -e "d$j\t\c" >> log/cufft.log
  for i in 0
  do
    cp data/d$j/fft$i.in fft.in
    echo case$i input_done

    nohup timeout 600 ./sfft >> log/sfft.log &
    echo case$i sfft_done
    nohup timeout 600 ./cufft >> log/cufft.log &
    echo case$i cufft_done 
    proc=1
    while(($proc<=6))
    do
      let "y=2**$proc"
      echo mpi $y
      mpiexec -n $y ./mpifft>> log/d$j/mpi.log
      echo -e "\t\c">> log/d$j/mpi.log
      proc=$((proc+1))
    done
    echo case$i mpifft_done
    echo -e "">> log/d$j/mpi.log
  done

  echo -e "" >> log/sfft.log
  echo -e "" >> log/cufft.log
done

for j in 26 27 28 29 30
do
  mkdir log/d$j
  date > log/d$j/mpi.log
  echo $j hahahah
  echo -e "d$j\t\c" >> log/sfft.log
  for i in 0
  do
    cp data/d$j/fft$i.in fft.in
    echo case$i input_done

    nohup timeout 300 ./sfft >> log/sfft.log &
    echo case$i sfft_done
    proc=4
    while(($proc<=7))
    do
      let "y=2**$proc"
      echo mpi $y
      mpiexec -n $y ./mpifft>> log/d$j/mpi.log
      echo -e "\t\c">> log/d$j/mpi.log
      proc=$((proc+1))
    done
    echo case$i mpifft_done
    echo -e "">> log/d$j/mpi.log
  done
  echo -e "" >> log/sfft.log
done
