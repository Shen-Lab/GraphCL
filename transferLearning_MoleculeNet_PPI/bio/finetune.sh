#### GIN fine-tuning
split=species

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --model_file models_graphcl/graphcl_80.pth --split $split --epochs 50 --device 0 --runseed $runseed --gnn_type gin --lr 1e-3
done
