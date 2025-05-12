import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
class DifBTrainer_wo_rtg:

    def __init__(self, frame_work, device, max_ep_len, sample_num, num_epochs, seq_len, goal_step, action_dim, beta, gamma, logger, env="PP4a"):
        self.frame_work = frame_work
        self.device = device
        self.max_ep_len = max_ep_len
        self.sample_num  = sample_num
        self.num_epochs = num_epochs
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.goal_step = goal_step
        self.gamma = gamma
        self.beta = beta
        self.logger = logger
        self.env = env

    def train(self, train_loader, epoch):
        self.frame_work.actor.model.train()
        self.frame_work.ReconGoal.train()
        self.frame_work.RtgNet.train()
        epoch_action_loss = 0.0
        epoch_rtg_loss = 0.0
        epoch_goal_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
            for batch_idx, episodes_data in enumerate(pbar):
                action_loss, rtg_loss, goal_loss = self.train_step(episodes_data)
                epoch_action_loss += action_loss
                epoch_goal_loss += goal_loss
                epoch_rtg_loss += rtg_loss
                pbar.set_postfix({
                    "Action Loss": f"{action_loss:.4f}",
                    "Rtg Loss": f"{rtg_loss:.4f}",
                    "Goal Loss": f"{goal_loss:.4f}",
                })

                # 日志记录
                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx}/{len(train_loader)}]\n"
                                f"Action Loss: {action_loss:.4f}\n"
                                f"Rtg Loss: {rtg_loss:.4f}\n" 
                                f"Goal Loss: {goal_loss:.4f}\n" 
                                )
        avg_epoch_action_loss = epoch_action_loss / len(train_loader)
        avg_epoch_rtg_loss = epoch_rtg_loss / len(train_loader)
        avg_epoch_goal_loss = epoch_goal_loss / len(train_loader)
        self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] completed. \n" 
                        f"Average Training Action Loss: {avg_epoch_action_loss :.4f}\n"
                        f"Average Training Rtg Loss: {avg_epoch_rtg_loss : .4f}\n"
                        f"Average Training Goal Loss: {avg_epoch_goal_loss :.4f}\n"
                        )
        self.logger.info("===================================================================================")
        return {"action_loss": avg_epoch_action_loss.detach().cpu().item(),
                "rtg_loss": avg_epoch_rtg_loss.detach().cpu().item(),
                "goal_loss": avg_epoch_goal_loss.detach().cpu().item(),
                }

    def step_loop(self, episodes_data, sample_list):
        action_loss = 0.0
        goal_loss = 0.0
        rtg_loss = 0.0
        
        for ts in sample_list:
            s = episodes_data["state"][:, ts, :, :].to(self.device)   # shape [batch, num, dim]
            g = episodes_data["state"][:, ts+self.goal_step, :, :].to(self.device)
            # o = episodes_data["obs"][:, ts, :].to(self.device)          # shape [batch, dim]
            o_seq = episodes_data["obs"][:, ts-self.seq_len+1:ts+1, :].to(self.device)  # shape [batch, seq, dim]
            s_seq = episodes_data["state"][:, ts-self.seq_len+1:ts+1, :, :].to(self.device)  # shape [batch, seq, num, dim]
            a_seq = episodes_data["action"][:, ts-self.seq_len:ts].to(self.device).unsqueeze(-1)  # shape [batch, seq, dim]
            a_seq = F.one_hot(a_seq.long().squeeze(-1), num_classes=self.action_dim).float()
            rtg = episodes_data["rtg"][:, ts].to(self.device).unsqueeze(-1)       # shape [batch, 1]
            action = episodes_data["action"][:, ts].to(self.device).unsqueeze(-1)
            t = torch.arange(0, self.seq_len).unsqueeze(0).expand(s.shape[0], self.seq_len).to(self.device)
            if self.env == "LBF":
                o_seq = F.one_hot(o_seq[...,[-2, -3]].long(), num_classes=20).float()
                o_seq = o_seq.view(*o_seq.shape[:-2], -1)
                s = F.one_hot(s[...,[-2, -3]].long(), num_classes=20).float()
                s = s.view(*s.shape[:-2], -1)
                s_seq = F.one_hot(s_seq[...,[-2, -3]].long(), num_classes=20).float()   
                s_seq = s_seq.view(*s_seq.shape[:-2], -1)
                g = F.one_hot(g[...,[-2, -3]].long(), num_classes=20).float()
                g = g.view(*g.shape[:-2], -1)
            z_mu, z_logvar = self.frame_work.StateEncoder(o_seq.unsqueeze(2), s_seq)
            z = torch.randn_like(z_mu) * torch.exp(z_logvar / 2) + z_mu
            # Convert action indices to one-hot vectors
            action_onehot = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()
            action = action_onehot
            ac_loss = self.frame_work.actor.loss(action, z)
            try:
                mid = self.frame_work.actor.model.get_mid()
            except:
                mid = self.frame_work.actor.model.module.get_mid()
            r_loss, g_loss = self.mid_loss(g, rtg, mid, s)
            action_loss = action_loss + ac_loss + self.beta * r_loss + self.gamma * g_loss
            rtg_loss += self.beta * r_loss 
            goal_loss += self.gamma * g_loss
        
        return action_loss, rtg_loss, goal_loss

    def train_step(self, episodes_data):
        sample_list = random.sample(range(self.seq_len + 1, self.max_ep_len - 1 - self.goal_step), self.sample_num)
        action_loss, rtg_loss, goal_loss = self.step_loop(episodes_data, sample_list)
        self.frame_work.actor_optimizer.zero_grad()
        action_loss.backward()
        self.frame_work.actor_optimizer.step()

        return action_loss / len(sample_list), rtg_loss / len(sample_list), goal_loss / len(sample_list)

    def mid_loss(self, g, rtg, mid, s):
        rtg_hat = self.frame_work.CoReturn(state=s, h=mid)
        g_hat = self.frame_work.CoGoal(state=s, r_hat=rtg_hat)
        rtg_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_fun = nn.BCELoss(reduction='mean')
        goal_loss = loss_fun(g_hat, g)
        return rtg_loss, goal_loss

    def evaluate(self, val_loader, epoch):
        self.frame_work.actor.model.eval()
        self.frame_work.CoGoal.eval()
        self.frame_work.CoReturn.eval()
        epoch_action_loss = 0.0
        epoch_goal_loss = 0.0
        epoch_rtg_loss = 0.0
        with torch.no_grad():
            for batch_idx, episodes_data in enumerate(val_loader):
                action_loss, rtg_loss, goal_loss= self.eval_step(episodes_data)
                epoch_action_loss += action_loss
                epoch_goal_loss += goal_loss
                epoch_rtg_loss += rtg_loss
        avg_epoch_action_loss = epoch_action_loss / len(val_loader)
        avg_epoch_goal_loss = epoch_goal_loss / len(val_loader)
        avg_epoch_rtg_loss = epoch_rtg_loss / len(val_loader)
        self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] Validation:\n"
                        f"Average Validation Action Loss: {avg_epoch_action_loss:.4f}\n"
                        f"Average Validation Goal Loss: {avg_epoch_goal_loss:.4f}\n"
                        f"Average Validation Rtg Loss: {avg_epoch_rtg_loss:.4f}\n")
        self.logger.info("===================================================================================")
        return {"action_loss": avg_epoch_action_loss.detach().cpu().item(),
                "goal_loss": avg_epoch_goal_loss.detach().cpu().item(),
                "rtg_loss": avg_epoch_rtg_loss.detach().cpu().item()
                }

    def eval_step(self, episodes_data):
        sample_list = random.sample(range(self.seq_len + 1, self.max_ep_len - 1 - self.goal_step), self.sample_num)
        action_loss, rtg_loss, goal_loss = self.step_loop(episodes_data, sample_list)
        return action_loss / len(sample_list), rtg_loss / len(sample_list), goal_loss / len(sample_list)

class DifBTrainer_wo_goal:

    def __init__(self, frame_work, device, max_ep_len, sample_num, num_epochs, seq_len, goal_step, action_dim, beta, gamma, logger, env="PP4a"):
        self.frame_work = frame_work
        self.device = device
        self.max_ep_len = max_ep_len
        self.sample_num  = sample_num
        self.num_epochs = num_epochs
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.goal_step = goal_step
        self.gamma = gamma
        self.beta = beta
        self.logger = logger
        self.env = env

    def train(self, train_loader, epoch):
        self.frame_work.actor.model.train()
        self.frame_work.ReconGoal.train()
        self.frame_work.RtgNet.train()
        epoch_action_loss = 0.0
        epoch_rtg_loss = 0.0
        epoch_goal_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
            for batch_idx, episodes_data in enumerate(pbar):
                action_loss, rtg_loss, goal_loss = self.train_step(episodes_data)
                epoch_action_loss += action_loss
                epoch_goal_loss += goal_loss
                epoch_rtg_loss += rtg_loss
 
                pbar.set_postfix({
                    "Action Loss": f"{action_loss:.4f}",
                    "Rtg Loss": f"{rtg_loss:.4f}",
                    "Goal Loss": f"{goal_loss:.4f}",
                })

                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx}/{len(train_loader)}]\n"
                                f"Action Loss: {action_loss:.4f}\n"
                                f"Rtg Loss: {rtg_loss:.4f}\n" 
                                f"Goal Loss: {goal_loss:.4f}\n" 
                                )
        avg_epoch_action_loss = epoch_action_loss / len(train_loader)
        avg_epoch_rtg_loss = epoch_rtg_loss / len(train_loader)
        avg_epoch_goal_loss = epoch_goal_loss / len(train_loader)

        self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] completed. \n" 
                        f"Average Training Action Loss: {avg_epoch_action_loss :.4f}\n"
                        f"Average Training Rtg Loss: {avg_epoch_rtg_loss : .4f}\n"
                        f"Average Training Goal Loss: {avg_epoch_goal_loss :.4f}\n"
                        )
        self.logger.info("===================================================================================")
        return {"action_loss": avg_epoch_action_loss.detach().cpu().item(),
                "rtg_loss": avg_epoch_rtg_loss.detach().cpu().item(),
                "goal_loss": avg_epoch_goal_loss.detach().cpu().item(),
                }

    def step_loop(self, episodes_data, sample_list):
        action_loss = 0.0
        goal_loss = 0.0
        rtg_loss = 0.0
        
        for ts in sample_list:
  
            s = episodes_data["state"][:, ts, :, :].to(self.device)   # shape [batch, num, dim]
            g = episodes_data["state"][:, ts+self.goal_step, :, :].to(self.device)
            # o = episodes_data["obs"][:, ts, :].to(self.device)          # shape [batch, dim]
            o_seq = episodes_data["obs"][:, ts-self.seq_len+1:ts+1, :].to(self.device)  # shape [batch, seq, dim]
            s_seq = episodes_data["state"][:, ts-self.seq_len+1:ts+1, :, :].to(self.device)  # shape [batch, seq, num, dim]
            a_seq = episodes_data["action"][:, ts-self.seq_len:ts].to(self.device).unsqueeze(-1)  # shape [batch, seq, dim]
            a_seq = F.one_hot(a_seq.long().squeeze(-1), num_classes=self.action_dim).float()
            rtg = episodes_data["rtg"][:, ts].to(self.device).unsqueeze(-1)       # shape [batch, 1]
            action = episodes_data["action"][:, ts].to(self.device).unsqueeze(-1)
            t = torch.arange(0, self.seq_len).unsqueeze(0).expand(s.shape[0], self.seq_len).to(self.device)
            if self.env == "LBF":
                o_seq = F.one_hot(o_seq[...,[-2, -3]].long(), num_classes=20).float()
                o_seq = o_seq.view(*o_seq.shape[:-2], -1)
                s_seq = F.one_hot(s_seq[...,[-2, -3]].long(), num_classes=20).float()   
                s_seq = s_seq.view(*s_seq.shape[:-2], -1)
                g = F.one_hot(g[...,[-2, -3]].long(), num_classes=20).float()
                g = g.view(*g.shape[:-2], -1)
            z_mu, z_logvar = self.frame_work.StateEncoder(o_seq.unsqueeze(2), s_seq)
            z = torch.randn_like(z_mu) * torch.exp(z_logvar / 2) + z_mu
            # Convert action indices to one-hot vectors
            action_onehot = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()
            action = action_onehot
            ac_loss = self.frame_work.actor.loss(action, z)
            try:
                mid = self.frame_work.actor.model.get_mid()
            except:
                mid = self.frame_work.actor.model.module.get_mid()
            r_loss, g_loss = self.mid_loss(g, rtg, mid, z)
            action_loss = action_loss + ac_loss + self.beta * r_loss + self.gamma * g_loss
            rtg_loss += self.beta * r_loss 
            goal_loss += self.gamma * g_loss
        
        return action_loss, rtg_loss, goal_loss

    def train_step(self, episodes_data):
        sample_list = random.sample(range(self.seq_len + 1, self.max_ep_len - 1 - self.goal_step), self.sample_num)
        action_loss, rtg_loss, goal_loss = self.step_loop(episodes_data, sample_list)
        self.frame_work.actor_optimizer.zero_grad()
        action_loss.backward()
        self.frame_work.actor_optimizer.step()

        return action_loss / len(sample_list), rtg_loss / len(sample_list), goal_loss / len(sample_list)

    def mid_loss(self, g, rtg, mid, z):
        rtg_hat = self.frame_work.RtgNet(torch.cat([z, mid], dim=-1))
        rtg_loss = F.mse_loss(rtg_hat, rtg)
        goal_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        return rtg_loss, goal_loss
    

    def evaluate(self, val_loader, epoch):
        self.frame_work.actor.model.eval()
        self.frame_work.ReconGoal.eval()
        self.frame_work.RtgNet.eval()
        epoch_action_loss = 0.0
        epoch_goal_loss = 0.0
        epoch_rtg_loss = 0.0
        with torch.no_grad():
            for batch_idx, episodes_data in enumerate(val_loader):
                action_loss, rtg_loss, goal_loss= self.eval_step(episodes_data)
                epoch_action_loss += action_loss
                epoch_goal_loss += goal_loss
                epoch_rtg_loss += rtg_loss
        avg_epoch_action_loss = epoch_action_loss / len(val_loader)
        avg_epoch_goal_loss = epoch_goal_loss / len(val_loader)
        avg_epoch_rtg_loss = epoch_rtg_loss / len(val_loader)
        self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] Validation:\n"
                        f"Average Validation Action Loss: {avg_epoch_action_loss:.4f}\n"
                        f"Average Validation Goal Loss: {avg_epoch_goal_loss:.4f}\n"
                        f"Average Validation Rtg Loss: {avg_epoch_rtg_loss:.4f}\n")
        self.logger.info("===================================================================================")
        return {"action_loss": avg_epoch_action_loss.detach().cpu().item(),
                "goal_loss": avg_epoch_goal_loss.detach().cpu().item(),
                "rtg_loss": avg_epoch_rtg_loss.detach().cpu().item()
                }

    def eval_step(self, episodes_data):
        sample_list = random.sample(range(self.seq_len + 1, self.max_ep_len - 1 - self.goal_step), self.sample_num)
        action_loss, rtg_loss, goal_loss = self.step_loop(episodes_data, sample_list)
        return action_loss / len(sample_list), rtg_loss / len(sample_list), goal_loss / len(sample_list)