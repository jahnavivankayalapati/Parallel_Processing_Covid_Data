#!/bin/bash

export MPI_COMM_WORLD_SIZE=32

# Run scripts and collect results
echo "Control"
python Control_OpenCovid.py > ../Results/open_control.txt
echo "MPI"
mpiexec -n $MPI_COMM_WORLD_SIZE python MPI_OpenCovid.py > ../Results/open_mpi.txt
echo "Multi"
python Multi_OpenCovid.py > ../Results/open_multi.txt