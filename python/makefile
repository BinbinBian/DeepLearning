
LAYERSIZES=500.1000.500
BATCHSIZE=-1




dbn:
	srun --comment="dbn_${LAYERSIZES}_${BATCHSIZE}" python DBN.py ${LAYERSIZES} ${BATCHSIZE} > log/dbn_${LAYERSIZES}_${BATCHSIZE}.txt
#-p long-sharedq 
