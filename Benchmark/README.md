# Superpixel datasets experiments
## 1. Pretraining:
For GIN on mnist dataset with drop node augmentation with projection head
`python main_superpixels_contrastive.py --config 1 --aug nn --head --gpu_id 0`
## 2. Finetuning:
For GIN on mnist dataset with drop node augmentation pre-trained model 
`python main_superpixels_graph_classification.py --config 1 --aug 0 --load_model --head --gpu_id 0`
