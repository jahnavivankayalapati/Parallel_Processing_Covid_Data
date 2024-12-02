#!/bin/bash

export MPI_COMM_WORLD_SIZE=32

echo "Control"
python Control_CovidDataset.py > ../Results/covidData_control.txt
echo "MPI"
mpiexec -n $MPI_COMM_WORLD_SIZE python MPI_CovidDataset.py > ../Results/covidData_mpi.txt
echo "Multi"
python Multi_CovidDataset.py > ../Results/covidData_multi.txt