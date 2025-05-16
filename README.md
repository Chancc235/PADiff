# PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork

This repository contains the implementation of PADiff, a Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork. Follow the instructions below to set up the environment and run the experiments.

## Quick Start

### Step 1: Setup Environment

First, navigate to the `src` directory and run the installation script:

```bash
cd src
bash install_env.sh
```

This script will install all the necessary dependencies including PyTorch, gym environments, and other required packages.

### Step 2: Train Teammates Policies

To train teammates policies for collaborative tasks, run:

```bash
bash run_train.sh
```

This script will train teammates policies for various environments. The trained models will be saved in the `saves` directory.

### Step 3: Collect Data

After training teammets policies, collect interaction data by running:

```bash
bash run_collect.sh
```

This script will use the trained partner policies to generate trajectories that will be used for training the PADiff model. The collected data will be saved in the `data` directory.

### Step 4: Train PADiff Model

Finally, navigate to the PADiff directory and run the training script:

```bash
cd PADiff
bash run.sh
```

This will train the PADiff model using the collected data. The script supports different environments which can be specified as command-line arguments.

## Advanced Usage

### Custom Environment Selection

You can train the PADiff model on a specific environment by specifying the environment name:

```bash
python train.py --env PP4a    # For Predator-Prey
python train.py --env LBF     # For Level-Based Foraging
python train.py --env overcooked  # For Overcooked
```

### Additional Options

- `--device gpu`: Use GPU
- `--seed 42`: Set random seed for reproducibility
- `--batch_size 128`: Override batch size from config
- `--epochs 20`: Override number of epochs from config

## Model Architecture

PADiff consists of several key components:

1. **AFM-Net**
2. **CoGoal**
3. **CoReturn**
4. **StateEncoder**

## Results

The model performance is logged in the following files:
- `loss.csv`: Training and validation losses
- `test_returns.csv`: Average returns during testing


## License

This project is licensed under the MIT License - see the LICENSE file for details.
