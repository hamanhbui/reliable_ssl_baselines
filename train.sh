# for i in {1..5}; do
#     python main.py --config "algorithms/ERM/configs/CIFAR10.json" --exp_idx $i --gpu_idx "0"
#     python main.py --config "algorithms/ERM/configs/CIFAR10_impulse_5.json" --exp_idx $i --gpu_idx "0"
# done

# python main.py --config "algorithms/ERM/configs/Rotated_75_MNIST.json" --exp_idx "1" --gpu_idx "1"
python main.py --config "algorithms/Jigsaw/configs/Rotated_75_MNIST.json" --exp_idx "1" --gpu_idx "1"