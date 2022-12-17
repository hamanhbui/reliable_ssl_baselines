# reliable_ssl_baselines

## Table of Content

1. [Introduction](#intro)
2. [Guideline](#guideline)
    - [To prepare](#prepare)
    - [To run experiments](#experiments)
    - [To plot figures](#plot)

## <a name="intro"></a> Introduction
![framework](gallery/framework.png)

This repository contains the implementation of 

## <a name="guideline"></a> Guideline
### <a name="prepare"></a> To prepare:
Install prerequisite packages:
```sh
python -m pip install -r requirements.txt
```

Download and unzip the datasets:
```sh
bash setup.sh
```

### <a name="experiments"></a> To run experiments:
Run with ten different seeds:
```sh
for i in {1..10}; do
     taskset -c <cpu_index> python main.py --config <config_path> --exp_idx $i --gpu_idx <gpu_index>
done
```
where the parameters are the following:
- `<cpu_index>`: CPU index. E.g., `<cpu_index> = "1"`
- `<config_path>`: path stored configuration hyper-parameters. E.g., `<config_path> = "algorithms/Jigsaw/configs/CIFAR10.json"`
- `<gpu_index>`: GPU index. E.g., `<gpu_index> = "0"`

**Note:** Select different settings by editing in `/configs/..json`, logging results are stored in `/results/logs/`

<img src="gallery/results.png" width="50%" height="50%">

### <a name="plot"></a> To plot feature representations:

```sh
python utils/ebar_plot.py
python utils/box_plot.py
python utils/plot_density.py
python utils/plot_density_methods.py
python utils/reliability_diagram.py
```

<img src="gallery/err_bar.png" width="50%" height="50%">
<img src="gallery/box_plot.png" width="50%" height="50%">
<img src="gallery/reliability_diagram.png" width="50%" height="50%">

## License

This source code is released under the Apache-2.0 license, included [here](LICENSE).