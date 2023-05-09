#!/bin/env bash

#SBATCH --job-name=skim
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --contiguous
#SBATCH --time=8:00:00
#SBATCH --output=out/%u_%x_%A.out
#SBATCH --mail-user=wl8015@princeton.edu
#SBATCH --mail-type=ALL

echo "directory: `pwd`"
echo "user:      `whoami`"
echo "host:      `hostname`"
cat /proc/$$/status | grep Cpus_allowed_list
echo ""

module load anacondapy/2022.05
conda activate sml505

python skim.py

exit 0
