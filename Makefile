.ONESHELL:

# ###############################################
# # Create environment and install dependencies #
# ###############################################
# create_environment:
# 	conda env create -f environment.yml
# 	source activate pass
# 	poetry install --no-root


############################################################
# Prepare data and ckpts to corresponding example folders ##
############################################################
data_assess_generators:
	cd assess_generators
	@echo ">>> Downloading generator checkpoints ..."
	gdown https://drive.google.com/drive/folders/1jrOyIxNsc4Wtdsv3LcJXLb8pF-VVBt6X?usp=drive_link --folder
	cd ..

data_multimodal:
	cd multimodal_inference
	python generate_imgs_multiargs.py
	cd ..


data_words_importance:
	cd words_importance

	mkdir -p ./data

	@echo ">>> Downloading imdb dataset to ./data/imdb_reviews.csv"
	gdown https://drive.google.com/file/d/188kXbMMJdP26A4_b_aH_iXHNoB45DZmW/view?usp=drive_link --fuzzy -O ./data/imdb_reviews.csv

	@echo ">>> Preparing sentiment word lists and pooled attention weights"
	python prepare.py
	
	@echo ">>> Downloading checkpoints for DistilBERT with positive/negative/netrual sentiment masking ..."
	gdown https://drive.google.com/drive/folders/1d_dYFOPwb0fbF4mEYYLwAl4XypLo3kbM?usp=drive_link -O ./ckpt/ --folder
	@echo ">>> Downloading checkpoints for affine coupling flow with positive/negative/netrual sentiment masking ..."
	gdown https://drive.google.com/drive/folders/1Sm60REyJmjaYtow78L3XxfM9QSJ7qCSN?usp=drive_link -O ./ckpt/ --folder
	
	@echo ">>> Downloading imdb embeddings with negative-sentiment-masking (neg) ..."
	gdown https://drive.google.com/drive/folders/13Ol-Q69Hnltp8__bmIY8HrU7jM8wftXU?usp=drive_link -O ./data/ --folder
	@echo ">>> Downloading imdb embeddings without any masking (none) ..."
	gdown https://drive.google.com/drive/folders/1Dz09_0t0GWm9BEUnSSH0VDWqxv85Oruq?usp=drive_link -O ./data/ --folder
	@echo ">>> Downloading imdb embeddings with neutral-sentiment-masking (others) ..."
	gdown https://drive.google.com/drive/folders/1cvkWDnXKRB0gG_z-1v0_xjAFK55HQInz?usp=drive_link -O ./data/ --folder
	@echo ">>> Downloading imdb embeddings with positive-sentiment-masking (pos) ..."
	gdown https://drive.google.com/drive/folders/1XIncd0nGYiMROT6GVb6K9vqWNfIfrssg?usp=drive_link -O ./data/ --folder
	
	cd ..
	




#######################################################
# Inference with prepared ckpts and null distribution #
#######################################################
inf_assess_generators:
	cd assess_generators
	
	python pai.py
	python visualization.py
	
	cd ..


inf_multimodal:
	cd multimodal_inference
	
	python pai_multiargs.py
	python visualization.py
		
	cd ..

inf_words_importance:
	cd words_importance
	
	python pai.py -m "W_pos_id"
	python pai.py -m "W_neg_id"
	python pai.py -m "W_others_id"
	
	python eval.py
	
	cd ..
	

#######################################################################################
# WIP: Inference start from scratch (time-consuming due to training and MC inference) #
#######################################################################################
