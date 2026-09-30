import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse

from Networks.ALTNet import ALT, TimestepEmbedder, ALTBlock, FinalLayer
from Networks.CoGoal import CoGoal
from Networks.CoReturn import CoReturn
from Networks.StateEncoder import StateEncoder

import os
import random
import csv
import time

from Data import CustomDataset
from Trainer import DifTrainer
from TestGame import Test
from Networks.Diffusion_models.PA_diffusion import PA_Diffusion
from utils_dt import load_config, create_save_directory, setup_logger, save_config
from Agent.DifAgent import DifAgent
import torch.multiprocessing as mp
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with environment selection')
    parser.add_argument('--env', type=str, choices=['PP4a', 'LBF', 'overcooked'], 
                        default='PP4a', help='Environment to use (PP4a, LBF, or overcooked)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=20, help='Override number of epochs from config')
    return parser.parse_args()

def test_base(test, agent, config, _):
    return test.test_game_dif(test_episodes=10, 
                             agent=agent, 
                             K=config["seq_len"])

def train_model(logger, trainer, train_loader, val_loader, num_epochs, device, test_interval, save_interval, save_dir, K, env_name, model_save_path="models", is_mp=False):
    if is_mp:
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    start_time = time.time()
    test = Test(env_name)
    
    val_csv_file_path = os.path.join(save_dir, 'loss.csv')
    with open(val_csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "val_action", "train_action", "val_rtg", "train_rtg", "val_goal", "train_goal",])

    for epoch in range(num_epochs):
        
        train_loss_dict = trainer.train(train_loader, epoch)
        val_loss_dict = trainer.evaluate(val_loader, epoch)

        with open(val_csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, val_loss_dict['action_loss'], train_loss_dict["action_loss"], 
                                        val_loss_dict['rtg_loss'], train_loss_dict["rtg_loss"], 
                                        val_loss_dict['goal_loss'], train_loss_dict["goal_loss"],
                                        ])
        end_time = time.time()
        epoch_duration = end_time - start_time
        hours, rem = divmod(epoch_duration, 3600)
        minutes, _ = divmod(rem, 60)
        logger.info(f"Completed in {int(hours)}h {int(minutes)}m")
        
        if (epoch + 1) % test_interval == 0 or epoch + 1 == 1:
            agent = DifAgent(trainer.frame_work, env_name)
            if is_mp:
                with mp.Pool(processes=5) as pool:
                    test_func = partial(test_base, test, agent, config)
                    results = pool.map(test_func, [None] * 5)
                    returns, variances = zip(*results)
                    returns = sum(returns) / len(returns)
                    var = sum(variances) / len(variances)
            else:
                returns, var = test.test_game_dif(50, agent, K)
            logger.info(f"{epoch + 1} Test Returns: {returns}")
            returns_csv_file_path = os.path.join(save_dir, 'test_returns.csv')
            with open(returns_csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, returns, var])
        
        if (epoch + 1) % save_interval == 0 or epoch + 1 == 1:
            dir_path = os.path.join(save_dir, model_save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_path = os.path.join(dir_path, f"epoch_{epoch+1}.pth") 
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': {
                    'model': trainer.frame_work.actor.model.state_dict(),
                    'StateEncoder': trainer.frame_work.StateEncoder.state_dict(),
                },
            }, save_path)
            
            logger.info(f"Model checkpoint saved at {save_path}")
            
    end_time = time.time()
    total_duration = end_time - start_time
    total_hours, total_rem = divmod(total_duration, 3600)
    total_minutes, _ = divmod(total_rem, 60)
    print(f"Training completed in {int(total_hours)}h {int(total_minutes)}m")

if __name__ == "__main__":

    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    env = args.env
    config = load_config(f"./config/{env}_config.yaml")
    
    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.epochs:
        config["num_epochs"] = args.epochs
    
    save_dir = create_save_directory()
    config["save_dir"] = save_dir
    logger = setup_logger(save_dir)

    logger.info(f"Using environment: {env}")

    # Initialize models
    StateEncoder = StateEncoder(state_dim=config["state_dim"],
                                num_agents=config["agent_num"],
                                embed_dim=config["embed_dim"],
                                seq_len=config["seq_len"],
                                num_heads=config["StateEncoder_num_heads"],
                                hidden_dim=config["StateEncoder_hidden_dim"]).to(config["device"])

    CoReturn_model = CoReturn(
        state_dim=config['state_dim'], 
        DiT_embed_dim=config["ALT_hidden_dim"], 
        num_agents=config["agent_num"], 
        hidden_dim=config["rtg_hidden_dim"]
    ).to(config["device"])

    CoGoal_model = CoGoal(
        hidden_dim=config["ReconGoal_hidden_dim"], 
        num=config["agent_num"], 
        state_dim=config["state_dim"]
    ).to(config["device"])

    alt_model = ALT(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        seq_len=config["seq_len"],
        agent_num=config["agent_num"]+1,
        hidden_size=config["ALT_hidden_dim"],
        depth=config["ALT_depth"],
        num_heads=config["ALT_heads"],
        embed_dim=config["embed_dim"],
        mlp_ratio=config["ALT_mlp_ratio"],
        dropout_rate=config["dropout_rate"]
    ).to(config["device"])
    
    # Initialize diffusion framework
    frame_work = PA_Diffusion(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        max_action=1,
        device=config["device"],
        discount=0.9,
        tau=0.8,
        lr=config["lr"],
        optimizer=config["optimizer"],
        model=alt_model,
        CoGoal=CoGoal_model,
        CoReturn=CoReturn_model,
        StateEncoder=StateEncoder,
        beta_min=config["beta_min"],
        beta_max=config["beta_max"],
        schedule_type=config["schedule_type"],
        n_timesteps=config["n_timesteps"],
        lambda_aux=config["lambda_aux"],
        cfg_scale=config["cfg_scale"],
    )
    
    # Initialize trainer
    trainer = DifTrainer(
        frame_work=frame_work,
        device=config["device"], 
        max_ep_len=config["max_ep_len"], 
        sample_num=config["sample_num"], 
        num_epochs=config["num_epochs"], 
        seq_len=config["seq_len"],
        goal_step=config["goal_step"],
        action_dim=config["action_dim"],
        beta=config["beta"],
        gamma=config["gamma"],
        logger=logger
    )

    # Save configuration
    save_config(config, save_dir)
    logger.info("Starting.")
    logger.info("Loading Data.")
    data_path = config["train_data_path"]
    
    # Log model parameters
    logger.info("Parameters num: ")
    logger.info(f"StateEncoder: {sum(p.numel() for p in StateEncoder.parameters())}")
    logger.info(f"Diffusion model: {sum(p.numel() for p in alt_model.parameters())}")
    logger.info(f"CoReturn: {sum(p.numel() for p in CoReturn_model.parameters())}")
    logger.info(f"CoGoal: {sum(p.numel() for p in CoGoal_model.parameters())}")
    
    # Load data
    load_start_time = time.time()
    data = torch.load(data_path)
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    hours, rem = divmod(load_duration, 3600)
    minutes, _ = divmod(rem, 60)
    logger.info(f"Data loaded in {int(hours)}h {int(minutes)}m")

    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=47)
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=0
    )

    # Start training
    logger.info("Training Started.")
    train_model(
        logger, 
        trainer,
        train_loader, 
        val_loader,
        num_epochs=config["num_epochs"], 
        device=config["device"], 
        test_interval=config["test_interval"],
        save_interval=config["save_interval"], 
        save_dir=config["save_dir"], 
        model_save_path=config["model_save_path"],
        K=config["seq_len"],
        env_name=env,
        is_mp=config["is_mp"]
    )
    
    logger.info("Training completed.")
