#!/bin/bash

top_k=600
threshold_proportion=0.02
data_folder="./data/"
ckpt_folder="./ckpt/ckpt_distilbert_att/"

mask_keyword_list=("W_pos_id" "W_neg_id" "W_others_id")


for mask_keyword in "${mask_keyword_list[@]}"; do
    python ./utils/train_distilbert.py --train -k "$top_k" -m "$mask_keyword" -tp "$threshold_proportion" --data_folder "$data_folder" --ckpt_folder "$ckpt_folder"
done