# root : 数据集安装路径
# dataset-config-file : 加载通常的数据设置
# config-file : 加载算法的超参数和优化参数
# output-dir : 结果输出路径

cd ..
DATA=~/kaiyang/data

# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root $DATA \
# --trainer Split \
# --source-domains cartoon photo sketch \
# --target-domains art_painting \
# --dataset-config-file configs/datasets/dg/pacs.yaml \
# --config-file configs/trainers/dg/myExp/pacs.yaml \
# --output-dir output/Split_globalAve

# On Pacs
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer Split \
--source-domains cartoon photo sketch \
--target-domains art_painting \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/myExp/pacs_split.yaml \
--output-dir output/test

# On office-home
# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root $DATA \
# --trainer Split \
# --source-domains art clipart product \
# --target-domains real_world \
# --dataset-config-file configs/datasets/dg/office_home_dg.yaml \
# --config-file configs/trainers/dg/myExp/office_home.yaml \
# --output-dir output/office_home/delete3

########### Multi-source #############
# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root $DATA \
# --trainer SourceOnly \
# --source-domains clipart painting real \
# --target-domains sketch \
# --dataset-config-file configs/datasets/da/mini_domainnet.yaml \
# --config-file configs/trainers/da/source_only/mini_domainnet.yaml \
# --output-dir output/source_only_minidn

########### Test #############
# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root $DATA \
# --trainer SourceOnly \
# --source-domains amazon \
# --target-domains webcam \
# --dataset-config-file configs/datasets/da/office31.yaml \
# --config-file configs/trainers/da/source_only/office31.yaml \
# --output-dir output/source_only_office31_test \
# --eval-only \
# --model-dir output/source_only_office31 \
# --load-epoch 20

