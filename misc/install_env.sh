#!/bin/bash
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --output=install_env_%j.log
export PIXI_CACHE_DIR=/work/users/s243203/.pixi-cache
cd /work/users/s243203/projects/pypsa-eur
pixi install