cd ..
DATA=~/kaiyang/data

# On office-home
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer Uncertainty \
--source-domains art clipart product \
--target-domains real_world \
--dataset-config-file configs/datasets/dg/office_home_dg.yaml \
--config-file configs/trainers/dg/myExp/office_home.yaml \
--output-dir output/office_home/delete3