for i in {2..5}; do
    python main.py --config "algorithms/Context/configs/CIFAR10.json" --exp_idx $i --gpu_idx "1"
done