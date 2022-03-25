cd ..
DATA=~/kaiyang/data

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer ResNet18 \
--source-domains cartoon sketch art_painting \
--target-domains photo \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/myExp/resnet_pacs.yaml \
--output-dir output/ResNet
