#!/bin/bash
export OMP_NUM_THREADS=2
/usr/local/shared/slurm/bin/srun -n 8 -m cyclic --mpi=pmi2 python3 MCMC.py
