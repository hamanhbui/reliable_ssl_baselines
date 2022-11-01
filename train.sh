# rm -r algorithms/Jigsaw/results/checkpoints/*
# rm -r algorithms/Jigsaw/results/logs/*
# rm -r algorithms/Jigsaw/results/plots/*
# rm -r algorithms/Jigsaw/results/tensorboards/*
# python main.py --config "algorithms/ERM/configs/CIFAR10.json" --exp_idx "1" --gpu_idx "1"
# python main.py --config "algorithms/Jigsaw/configs/CIFAR10.json" --exp_idx "1" --gpu_idx "1"
python main.py --config "algorithms/Rotation/configs/CIFAR10.json" --exp_idx "1" --gpu_idx "0"