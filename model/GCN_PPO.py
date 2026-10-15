from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import time

torch.autograd.set_detect_anomaly(True)
# class GCNLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#     def forward(self, x, adj):
#         # x: [B, N, F], adj: [B, N, N]

#         I = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)  # [1, N, N]
#         adj_with_self_loops = adj + I  # [B, N, N]

#         degree = torch.sum(adj_with_self_loops, dim=2)  # [B, N]
#         D_inv_sqrt = torch.pow(degree, -0.5)  # [B, N]

#         # Step 3: 构建 D^(-1/2) * A * D^(-1/2)
#         D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # [B, N, N]
#         A_hat = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_with_self_loops), D_inv_sqrt_mat)  # [B, N, N]

#         # Step 4: 聚合邻居特征
#         h = torch.bmm(A_hat, x)  # [B, N, F_in]
#         h = self.linear(h)       # [B, N, F_out]

#         return h

# class GCNActorNetwork(nn.Module):
#     def __init__(self, feature_dim, input_dim):
#         super().__init__()
#         self.gcn1 = GCNLayer(feature_dim, 64)
#         self.norm1 = nn.LayerNorm(64)
#         self.gcn2 = GCNLayer(64, 32)
#         self.norm2 = nn.LayerNorm(32)
#         self.gcn3 = GCNLayer(32, 1)
#         self.input_dim = input_dim

#     def forward(self, x, adj):
#         # x: [B, N, F], adj: [B, N, N] (batch-wise dense)
#         x = F.leaky_relu(self.gcn1(x, adj)) # [B, N, 64]
#         x = self.norm1(x)                    # [B, N, 64]
#         x = F.leaky_relu(self.gcn2(x, adj)) # [B, N, 32]
#         x = self.norm2(x)                    # [B, N, 32]
#         x = self.gcn3(x, adj)               # [B, N, 1]
#         x = x.squeeze(-1)  # [B, N]
#         # print("min x: ", torch.min(x), "max x: ", torch.max(x))
#         return x

# class GCNCriticNetwork(nn.Module):
#     def __init__(self, feature_dim, input_dim):
#         super().__init__()
#         self.gcn1 = GCNLayer(feature_dim, 64)
#         self.norm1 = nn.LayerNorm(64)
#         self.gcn2 = GCNLayer(64, 32)
#         self.norm2 = nn.LayerNorm(32)
#         self.input_dim = input_dim
#         self.fc = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.LeakyReLU(),
#             nn.Linear(16, 1)
#         )

#     def forward(self, x, adj):
#         # x: [B, N, F], adj: [B, N, N]
#         x = F.leaky_relu(self.gcn1(x, adj))
#         x = self.norm1(x)
#         x = F.leaky_relu(self.gcn2(x, adj))
#         x = self.norm2(x)
#         x = x.mean(dim=1)  # [B, 32]
#         v = self.fc(x)  # [B, 1]
#         return v.squeeze()

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, adj):
        # x: [B, N, F], adj: [B, N, N]

        I = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)  # [1, N, N]
        adj_with_self_loops = adj + I  # [B, N, N]

        degree = torch.sum(adj_with_self_loops, dim=2)  # [B, N]
        D_inv_sqrt = torch.pow(degree, -0.5)  # [B, N]

        # Step 3: 构建 D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # [B, N, N]
        A_hat = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_with_self_loops), D_inv_sqrt_mat)  # [B, N, N]

        # Step 4: 聚合邻居特征
        h = torch.bmm(A_hat, x)  # [B, N, F_in]
        h = self.linear(h)       # [B, N, F_out]
        # print("contduct x: ", h.shape, "adj: ", adj.shape, "D_inv_sqrt: ", D_inv_sqrt.shape, "A_hat: ", A_hat.shape)

        return h

class GCNActorNetwork(nn.Module):
    def __init__(self, feature_dim, input_dim):
        super().__init__()
        self.gcn1 = GCNLayer(feature_dim, 64)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.gcn2 = GCNLayer(64, 32)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(32)
        self.gcn3 = GCNLayer(32, 1)
        self.input_dim = input_dim

    def forward(self, x, adj):
        # x: [B, N, F], adj: [B, N, N] (batch-wise dense)
        x = self.norm1(x)                    # [B, N, 64]
        x = F.leaky_relu(self.gcn1(x, adj)) # [B, N, 64]
        x = self.norm2(x)                    # [B, N, 32]
        x = F.leaky_relu(self.gcn2(x, adj)) # [B, N, 32]
        x = self.norm3(x)                    # [B, N, 32]
        x = self.gcn3(x, adj)               # [B, N, 1]
        x = x.squeeze(-1)  # [B, N]
        # print("min x: ", torch.min(x), "max x: ", torch.max(x))
        return x

class GCNCriticNetwork(nn.Module):
    def __init__(self, feature_dim, input_dim):
        super().__init__()
        self.gcn1 = GCNLayer(feature_dim, 64)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.gcn2 = GCNLayer(64, 32)
        self.norm2 = nn.LayerNorm(64)
        self.gcn3 = GCNLayer(32, 1)
        self.norm3 = nn.LayerNorm(32)
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, adj):
        # x: [B, N, F], adj: [B, N, N]
        x = self.norm1(x)
        x = F.leaky_relu(self.gcn1(x, adj))
        x = self.norm2(x)
        x = F.leaky_relu(self.gcn2(x, adj))
        x = self.norm3(x)
        x = F.leaky_relu(self.gcn3(x, adj))  # [B, N, 1]
        x = x.squeeze(-1)  # [B, N]
        # x = x.mean(dim=1)  # [B, 32]
        v = self.fc(x)  # [B, 1]
        return v.squeeze()

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class GCN_PPO:
    def __init__(self, feature_dim, input_dim, config, get_adjacency_fn):
        self.feature_dim = feature_dim
        self.model = config['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config[self.model]['gamma']
        self.lam = config[self.model]['lambda']
        self.clip_epsilon = config[self.model]['clip_epsilon']
        self.ppo_epochs = config[self.model]['ppo_epochs']
        self.mini_batch_size = config[self.model]['mini_batch_size']
        self.batch_size = config[self.model]['batch_size']
        self.actor_lr = config[self.model]['actor_lr']
        self.critic_lr = config[self.model]['critic_lr']
        self.entropy_coef = config[self.model]['entropy_coef']
        self.max_grad_norm = config[self.model]['max_grad_norm']
        
        # Networks
        self.actor = GCNActorNetwork(feature_dim, input_dim).to(self.device)
        self.critic = GCNCriticNetwork(feature_dim, input_dim).to(self.device)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.get_adjacency_fn = get_adjacency_fn
        print(f"Actor network: {self.actor}")
        print(f"Critic network: {self.critic}")
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Tracking
        self.episode_losses = []
        self.actor_losses = []
        self.critic_losses = []
        
        # PPO memory (for current trajectory only)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
        self.values = []
        self.next_values = []
        self.mask = []  # Mask for valid actions, used in choose_node

    def reset(self):
        self.episode_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.clear_trajectory()
        
    def clear_trajectory(self):
        """Clear the current trajectory data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []
        self.values = []
        self.next_values = []
        self.mask = []

    def remember(self, state, mask, action, reward, next_state, done, action_prob):
        """Store a transition in the current trajectory"""
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu()
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.action_probs.append(action_prob)
        self.mask.append(mask)

    def save(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved actor to {actor_path} and critic to {critic_path}")

    def load(self, actor_path='ppo_actor.pth', critic_path='ppo_critic.pth'):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor.eval()
        self.critic.eval()
        print(f"Loaded actor from {actor_path} and critic from {critic_path}")

    def get_feature_matrix(self, sfcr, candidates_sn, network, vn):
        feature_matrix = network.get_feature_matrix(sfcr, candidates_sn, vn)
        return torch.tensor(feature_matrix, dtype=torch.float32) / torch.tensor([100., 100., 100., 1., 1.])

    def select_action(self, sfcr, candidates_sn, vn, network, mask):
        feature_matrix = self.get_feature_matrix(sfcr, candidates_sn, network, vn)
        feature_tensor = feature_matrix.unsqueeze(0).to(self.device).clone()
        adj = torch.tensor(self.get_adjacency_fn(network), dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, N, N]
        with torch.no_grad():
            logits = self.actor(feature_tensor, adj).squeeze()  # [input_dim]
            logits = logits.masked_fill(mask.to(self.device)==False, float('-inf'))  # Apply mask to logits
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample().item()
            action_log_prob = probs.log_prob(torch.tensor(action).to(self.device)).item()
        return action, feature_tensor, action_log_prob
    
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                gae = 0
                next_val = 0
            else:
                next_val = next_values[i]
            delta = rewards[i] + self.gamma * (1 - dones[i]) * next_val - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        # Convert to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_policy(self):
        # If not enough experiences, skip update
        if len(self.states) < self.batch_size:
            return
        # Process trajectory data
        states = torch.stack(self.states).to(self.device).squeeze()
        masks = torch.stack(self.mask).to(self.device).squeeze()
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        old_log_action_probs = torch.tensor(self.action_probs, dtype=torch.float32).to(self.device)
        next_states = torch.stack(self.next_states).to(self.device).squeeze()
        batch_adj = torch.tensor(self.get_adjacency_fn(), dtype=torch.float32).unsqueeze(0).repeat(states.shape[0], 1, 1).to(self.device)
        values = self.critic(states, batch_adj).to(self.device)
        next_values = self.critic(next_states, batch_adj).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values.detach(), dones, next_values.detach())
        
        # Prepare for mini-batch updates
        batch_size = min(len(self.states), self.batch_size)

        # PPO multiple epochs
        _loss = {
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': []
        }

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > batch_size:
                    end = batch_size
                idx = indices[start:end]
                
                mini_states = states[idx]
                mini_actions = actions[idx]
                mini_old_log_action_probs = old_log_action_probs[idx]
                mini_advantages = advantages[idx]
                mini_returns = returns[idx]
                mini_adj = batch_adj[idx]
                mini_masks = masks[idx]

                # Calculate new probabilities and values
                logits = self.actor(mini_states, mini_adj)  # [mini_batch_size, input_dim]
                # print('logits shape: ', logits.shape)
                # print('mask shape: ', mini_masks.shape)
                logits = logits.masked_fill(mini_masks.to(self.device)==False, float('-inf'))  # Apply mask to logits
                new_probs = torch.distributions.Categorical(logits=logits) # [mini_batch_size, input_dim]
                new_log_action_probs = new_probs.log_prob(mini_actions)  # [mini_batch_size]
                # print(mini_actions.shape, new_probs.probs.shape)
                new_action_probs = new_probs.probs.gather(1, mini_actions.unsqueeze(1)).squeeze(1)  # [mini_batch_size]
                # PPO clipped objective
                ratio = torch.exp(new_log_action_probs - mini_old_log_action_probs)                         # [mini_batch_size]
                
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_advantages
                actor_loss = torch.mean(-torch.min(surr1, surr2))
                
                # Critic loss
                critic_loss = F.mse_loss(self.critic(mini_states, mini_adj), mini_returns.detach())

                # Entropy bonus
                assert new_action_probs.shape == new_log_action_probs.shape, "Shape mismatch between action probabilities and log probabilities"
                entropy = torch.mean(-torch.sum(new_action_probs * new_log_action_probs, dim=-1))

                # check NaN values for entropy
                if torch.isnan(entropy).any():
                    print("Entropy NaN detected")
                    print("Entropy: ", entropy)
                    print("New probs: ", new_log_action_probs)
                    print("Old probs: ", mini_old_log_action_probs)
                    print("Ratio: ", ratio)
                    print("Advantages: ", mini_advantages)
                    print("Returns: ", mini_returns)
                    print("Critic loss: ", critic_loss.item())
                    print("Actor loss: ", actor_loss.item())

                # Total loss
                loss = actor_loss + critic_loss - self.entropy_coef * entropy

                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {loss.item():.4f}")

                _loss['actor_loss'].append(actor_loss.item())
                _loss['critic_loss'].append(critic_loss.item())
                _loss['total_loss'].append(loss.item())

        self.clear_trajectory()

        return {key: np.mean(value) for key, value in _loss.items()}