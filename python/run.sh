
LAYERSIZES=1000
BATCHSIZE=-1

COMMENT=DBN_${LAYERSIZES}_${BATCHSIZE}

srun -p long-sharedq --comment="${COMMENT}" python DBN.py ${LAYERSIZES} ${BATCHSIZE} > log/${COMMENT}.txt

