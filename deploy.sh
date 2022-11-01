for i in {7..10}; do
    python main.py --config "algorithms/ERM/configs/CIFAR10.json" --exp_idx $i --gpu_idx "1"
done