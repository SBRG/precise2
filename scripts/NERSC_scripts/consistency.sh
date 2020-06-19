#!/bin/bash

NUM_RUNS=10
ITER=100
FILE="log_tpm_norm.csv"
TOL="1e-7"
NODES=1
TASKS=32
TIME="00:30:00"
EMAIL="avsastry@eng.ucsd.edu"

NTASKS=$(( NODES * TASKS ))

dir_name=${PWD##*/}

TMP_DIR="$SCRATCH/$dir_name"

if [ ! -f $FILE ]; then
    echo "File $FILE not found!"
    exit 1
fi

for i in  $(seq 1 $NUM_RUNS)
do

run_script="#!/bin/bash -l
\n
\n#SBATCH -p regular
\n#SBATCH -N $NODES --ntasks-per-node=$TASKS
\n#SBATCH -t $TIME
\n#SBATCH --mail-user=$EMAIL
\n#SBATCH --mail-type=ALL
\n#SBATCH -C haswell
\n
\nmodule load python
\n
\nsrun -n $NTASKS python ../random_restart_ica.py -f $FILE -i $ITER -t $TOL -x $TMP_DIR/$i
\nsrun -n $NTASKS python ../compute_distance.py -i $ITER -x $TMP_DIR/$i
\nsrun -n 1 -N 1 python ../cluster_components.py -i $ITER -x $TMP_DIR/$i -w $NTASKS
"

    echo "Running iteration $i"
    mkdir $i
    cp $FILE $i
    echo -en $run_script > $i/${dir_name}_$i
    cd $i
    sbatch ${dir_name}_$i
    cd ..
done
