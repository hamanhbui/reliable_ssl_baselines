for i in {6..10}; do
    python main.py --config "algorithms/Jigsaw/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
done