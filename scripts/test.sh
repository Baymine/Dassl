cd ..
DATA=~/kaiyang/data

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

for SEED in $(seq 1 6)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        CUDA_VISIBLE_DEVICES=0 python tools/train.py \
        --root $DATA \
        --seed ${SEED} \
        --trainer DDAIG \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/dg/pacs.yaml \
        --config-file configs/trainers/dg/ddaig/pacs.yaml \
        --output-dir output/${DATASET}/DDAIG/${T}/seed${SEED}
    done
done