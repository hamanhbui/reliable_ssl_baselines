for i in {1..10}; do
    python main.py --config "algorithms/Context/configs/CIFAR10.json" --exp_idx $i --gpu_idx "0"
done