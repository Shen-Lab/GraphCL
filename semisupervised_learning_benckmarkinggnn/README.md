# Superpixel datasets experiments
## 1. Requirements
Please follow this [repo](https://github.com/graphdeeplearning/benchmarking-gnns) to create your environment and download datasets.

## 2. Pretraining:
For GIN on mnist dataset with drop node augmentation with projection head

`python main_superpixels_contrastive.py --config 1 --aug nn --head --gpu_id 0`
## 3. Finetuning:
For GIN on mnist dataset with drop node augmentation pre-trained model 

`python main_superpixels_graph_classification.py --config 1 --aug 0 --load_model --head --gpu_id 0`
