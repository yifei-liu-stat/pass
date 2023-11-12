# #!/bin/bash

# top_k=600
# threshold_value=0.02
# data_folder="./data/"
# ckpt_folder="./ckpt/ckpt_distilbert_att/"

# # python imdb_distillbert_lit_att.py --train -k $top_k

# python ./utils/train_distilbert.py --train -k $top_k -m W_pos_id -tp $threshold_value --data_folder $data_folder --ckpt_folder $ckpt_folder

# python ./utils/train_distilbert.py --train -k $top_k -m W_neg_id -tp $threshold_value --data_folder $data_folder --ckpt_folder $ckpt_folder

# python ./utils/train_distilbert.py --train -k $top_k -m W_others_id -tp $threshold_value --data_folder $data_folder --ckpt_folder $ckpt_folder



#!/bin/bash

top_k=600
threshold_value=0.02
data_folder="./data/"
ckpt_folder="./ckpt/ckpt_distilbert_att/"

mask_keyword_list=("W_pos_id" "W_neg_id" "W_others_id")


for mask_keyword in "${mask_keyword_list[@]}"; do
    python ./utils/train_distilbert.py --train -k "$top_k" -m "$mask_keyword" -tp "$threshold_value" --data_folder "$data_folder" --ckpt_folder "$ckpt_folder"
done