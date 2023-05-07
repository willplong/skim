#!/bin/env bash

#SBATCH --job-name=test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --contiguous
#SBATCH --time=0:30:00
#SBATCH --output=logs/%u_%x_%A.out
#SBATCH --mail-user=sy9959@princeton.edu
#SBATCH --mail-type=END

echo "directory: `pwd`"
echo "user:      `whoami`"
echo "host:      `hostname`"
cat /proc/$$/status | grep Cpus_allowed_list
echo ""

module load anacondapy/2020.11
source activate sml505

python /jukebox/witten/Sae/to_delete/skim.py

exit 0