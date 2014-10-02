LAYERSIZES=500.500
NUMTRAIN=5000
NUMTEST=1000


ALGO_HOME_FOLDER=/home_bunch/dongykan/workspace/DeepLearning/cpp/
DATA_HOME_FOLDER=/home_bunch/dongykan/workspace/DeepLearning/data/
DATASET=mnist
ALGO=test

LOG_FOLDER=log/

CMD=""
CMD="${ALGO_HOME_FOLDER}${ALGO} ${DATA_HOME_FOLDER}${DATASET}/ ${NUMTRAIN} ${NUMTEST} ${LAYERSIZES}" 
CMD="srun -p long-sharedq --comment='${ALGO}.${DATASET}.${NUMTRAIN}.${NUMTEST}.${LAYERSIZES}' $CMD"
LOG_FILENAME=${LOG_FOLDER}${ALGO}.${DATASET}.${NUMTRAIN}.${NUMTEST}.${LAYERSIZES}.log
CMD="$CMD > ${LOG_FILENAME}"

echo "=============================================="
echo $CMD
echo "=============================================="
eval $CMD



