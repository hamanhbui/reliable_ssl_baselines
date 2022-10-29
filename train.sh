# rm -r algorithms/Jigsaw/results/checkpoints/*
# rm -r algorithms/Jigsaw/results/logs/*
# rm -r algorithms/Jigsaw/results/plots/*
# rm -r algorithms/Jigsaw/results/tensorboards/*
# for i in {1..5}; do
#     python main.py --config "algorithms/ERM/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
# done
# python main.py --config "algorithms/Jigsaw/configs/CIFAR10.json" --exp_idx "1" --gpu_idx "1"
python main.py --config "algorithms/Jigsaw/configs/MNIST.json" --exp_idx "5" --gpu_idx "1"