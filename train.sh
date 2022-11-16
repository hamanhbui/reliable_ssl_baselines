# rm -r algorithms/Jigsaw/results/checkpoints/*
# rm -r algorithms/Jigsaw/results/logs/*
# rm -r algorithms/Jigsaw/results/plots/*
# rm -r algorithms/Jigsaw/results/tensorboards/*
# python main.py --config "algorithms/ERM/configs/CIFAR10_skew_1.json" --exp_idx "10" --gpu_idx "1"
# python main.py --config "algorithms/Rotation/configs/CIFAR10_skew_1.json" --exp_idx "1" --gpu_idx "1"
for i in {1..10}; do
    python main.py --config "algorithms/Affine/configs/CIFAR10_skew_4.json" --exp_idx $i --gpu_idx "0"
done
# python main.py --config "algorithms/ERM/configs/CIFAR10_skew_5.json" --exp_idx "4" --gpu_idx "0"
