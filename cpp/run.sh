#!/bin/sh

NUMTRAIN=500
NUMTEST=500
LAYERSIZE=100
BATCHSIZE=50 #0(full), 1, 2, ...N
MKL=false
THREAD=false
NUMTHREAD=4


ALGO_HOME_FOLDER=/home_bunch/dongykan/workspace/DeepLearning/cpp/
DATA_HOME_FOLDER=/home_bunch/dongykan/workspace/DeepLearning/data/
DATASET=mnist
ALGO=test
LOG_FOLDER=log/



NUMTRAIN=500
NUMTEST=100
LAYERSIZES=(500) # 1000) # 1000)
BATCHSIZES=(50) # 100) # 50 100)
MKLS=(true) # false true)
THREADS=(true)

for LAYERSIZE in "${LAYERSIZES[@]}" 
do
	for BATCHSIZE in "${BATCHSIZES[@]}"
	do
		for MKL in "${MKLS[@]}"
		do
			for THREAD in "${THREADS[@]}"
			do
				CMD=""
				CMD="${ALGO_HOME_FOLDER}${ALGO} ${DATA_HOME_FOLDER}${DATASET}/ ${NUMTRAIN} ${NUMTEST} ${LAYERSIZE} ${BATCHSIZE} ${MKL} ${THREAD} ${NUMTHREAD}" 
				CMD="srun -p long-sharedq --comment='${ALGO}.${DATASET}.${NUMTRAIN}.${NUMTEST}.${LAYERSIZE}_${BATCHSIZE}_${MKL}_${THREAD}.${NUMTHREAD}' $CMD"
				LOG_FILENAME=${LOG_FOLDER}${ALGO}.${DATASET}.${NUMTRAIN}.${NUMTEST}.${LAYERSIZE}_${BATCHSIZE}_${MKL}_${THREAD}.${NUMTHREAD}.log
				CMD="$CMD > ${LOG_FILENAME}"
				echo "=============================================="
				echo $CMD
				echo "=============================================="
				eval $CMD &
			done
		done
	done
	
done





