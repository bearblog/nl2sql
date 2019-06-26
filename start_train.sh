# CUDA_VISIBLE_DEVICES=$1 python train.py --ca --gpu --bs $2
# sample data
# CUDA_VISIBLE_DEVICES=$1 python train.py --ca --toy --gpu --bs $2 

#main
# CUDA_VISIBLE_DEVICES=1 python train.py --ca --toy 
# python main.py --ca --toy --gpu
python main.py --ca --gpu --nick $1
