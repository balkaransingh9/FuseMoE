import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse

# -------------------------
# 1. Synthetic Data Generation
# -------------------------
def generate_hierarchical_data(N=1000, noise_std=0.1, random_seed=42):
    np.random.seed(random_seed)
    X = np.random.rand(N, 2)
    y = np.zeros(N)
    expert_labels = np.zeros(N, dtype=int)
    
    for i in range(N):
        x0, x1 = X[i]
        if x0 < 0.5:
            # Branch 1
            if x1 < 0.5:
                y[i] = np.sin(2 * np.pi * x0) + np.random.normal(0, noise_std)
                expert_labels[i] = 1
            else:
                y[i] = np.cos(2 * np.pi * x1) + np.random.normal(0, noise_std)
                expert_labels[i] = 2
        else:
            # Branch 2
            if x1 < 0.5:
                y[i] = (x0 ** 2) + np.random.normal(0, noise_std)
                expert_labels[i] = 3
            else:
                y[i] = 2 * (x1 - x0) + np.random.normal(0, noise_std)
                expert_labels[i] = 4
                
    return X, y, expert_labels

# -------------------------
# 2. Model Definitions
# -------------------------
# 2.1 One-Level MoE Model
class OneLevelMoE(nn.Module):
    def __init__(self, input_dim=2, num_experts=4, hidden_dim=16):
        super(OneLevelMoE, self).__init__()
        self.num_experts = num_experts
        
        # Gating network: outputs weights for each expert
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Experts: each expert outputs a scalar prediction
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        gate_weights = self.gate(x)  # Shape: [batch_size, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [batch, 1, num_experts]
        # Weighted sum over experts:
        output = torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)
        return output, gate_weights

# 2.2 Two-Level HME Model
class TwoLevelHME(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super(TwoLevelHME, self).__init__()
        # First-level gating: splits based on coarse feature (assume two branches)
        self.first_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Two branches
            nn.Softmax(dim=1)
        )
        # Second-level experts for Branch 1 (2 experts) and Branch 2 (2 experts)
        self.branch1_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.branch2_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Experts for each branch: 2 experts per branch
        self.branch1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(2)
        ])
        self.branch2_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(2)
        ])
        
    def forward(self, x):
        # First-level gating (coarse split)
        first_weights = self.first_gate(x)  # [batch, 2]
        
        # For simplicity, split the batch into two groups based on the first feature threshold (or use soft decisions)
        # Here, we use the gating network outputs to weight contributions from branch experts.
        # Compute branch-specific outputs:
        
        # Branch 1 processing
        branch1_gate_weights = self.branch1_gate(x)  # [batch, 2]
        branch1_expert_outs = torch.stack([expert(x) for expert in self.branch1_experts], dim=2)  # [batch, 1, 2]
        branch1_output = torch.bmm(branch1_expert_outs, branch1_gate_weights.unsqueeze(2)).squeeze(2)  # [batch, 1]
        
        # Branch 2 processing
        branch2_gate_weights = self.branch2_gate(x)  # [batch, 2]
        branch2_expert_outs = torch.stack([expert(x) for expert in self.branch2_experts], dim=2)  # [batch, 1, 2]
        branch2_output = torch.bmm(branch2_expert_outs, branch2_gate_weights.unsqueeze(2)).squeeze(2)  # [batch, 1]
        
        # Combine branch outputs using the first-level gating weights:
        # first_weights[:,0] corresponds to branch1 and first_weights[:,1] to branch2.
        output = first_weights[:, 0:1] * branch1_output + first_weights[:, 1:2] * branch2_output
        
        # For analysis, return all gating weights
        gating_info = {
            'first_level': first_weights,
            'branch1': branch1_gate_weights,
            'branch2': branch2_gate_weights
        }
        return output, gating_info

# -------------------------
# 3. Training and Evaluation Functions
# -------------------------
def train_model(model, train_loader, num_epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred, _ = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_vals = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred, gating_info = model(x_batch)
            predictions.append(y_pred)
            true_vals.append(y_batch)
    predictions = torch.cat(predictions, dim=0)
    true_vals = torch.cat(true_vals, dim=0)
    mse = nn.MSELoss()(predictions, true_vals)
    return mse.item(), predictions, true_vals

# -------------------------
# 4. Main Experiment
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic experiment for HMoE")
    parser.add_argument("--test_rs", type=int, default=1, help="random seed for test data")
    parser.add_argument("--test_noise", type=float, default=0.1, help="noise magnitude for tesrt data")
    args = parser.parse_args()

    # Generate train and test sets
    test_rs = args.test_rs
    test_noise = args.test_noise
    X_train, y_train, labels_train = generate_hierarchical_data(N=2000, noise_std=0.1, random_seed=42)
    X_test, y_test, labels_test = generate_hierarchical_data(N=500, noise_std=test_noise, random_seed=test_rs)

    # Visualize the input space colored by the expert that generated each point.
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_test, cmap='viridis', alpha=0.7)
    for label in np.unique(labels_test):
        idx = labels_test == label
        plt.scatter(X_test[idx, 0], X_test[idx, 1], label=f'Expert {label}', alpha=0.7)
    plt.xlabel('$x_0$', fontsize=25)
    plt.ylabel('$x_1$', fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.title('Expert Assignment in Input Space', fontsize=20)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/sample_plot_1_{test_rs}_{test_noise}.pdf')

    # Visualize the target function behavior by plotting y against x0,
    # separately for each expert.
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels_test):
        idx = labels_test == label
        plt.scatter(X_test[idx, 0], y_test[idx], label=f'Expert {label}', alpha=0.7)
    plt.xlabel('$x_0$', fontsize=25)
    plt.ylabel('Target $y$', fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.title('Target Function Values vs $x_0$ by Expert', fontsize=25)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/sample_plot_2_{test_rs}_{test_noise}.pdf')

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Instantiate models
    model_one_level = OneLevelMoE(input_dim=2, num_experts=4, hidden_dim=16)
    model_two_level = TwoLevelHME(input_dim=2, hidden_dim=16)
    
    print("Training One-Level MoE:")
    model_one_level = train_model(model_one_level, train_loader, num_epochs=100, learning_rate=0.01)
    mse_one_level, preds_one, true_vals = evaluate_model(model_one_level, test_loader)
    print(f"One-Level MoE Test MSE: {mse_one_level:.4f}")
    
    print("\nTraining Two-Level HME:")
    model_two_level = train_model(model_two_level, train_loader, num_epochs=100, learning_rate=0.01)
    mse_two_level, preds_two, _ = evaluate_model(model_two_level, test_loader)
    print(f"Two-Level HME Test MSE: {mse_two_level:.4f}")
    mse_diff = (mse_one_level - mse_two_level) / mse_one_level
    # -------------------------
    # 5. Visualization of Gating Outputs (for Two-Level HME)
    # -------------------------
    # Visualize the first-level gating outputs on test set
    model_two_level.eval()
    with torch.no_grad():
        _, gating_info = model_two_level(X_test_tensor)
        first_level_weights = gating_info['first_level'].numpy()
        branch1_weights = gating_info['branch1'].numpy()
        branch2_weights = gating_info['branch2'].numpy()
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=first_level_weights[:, 0], cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("First-Level Weight (Branch 1)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("$x_0$", fontsize=25)
    plt.ylabel("$x_1$", fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.title("HMoE: First-Level Gating Weights", fontsize=21)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/weights_{mse_diff}_{test_rs}_{test_noise}.pdf')

    # Visualize Branch 1 gating weights for Expert 1
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=branch1_weights[:, 0], cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Branch 1 - Expert 1 Weight", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("$x_0$", fontsize=25)
    plt.ylabel("$x_1$", fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.title("HMoE: Branch 1 Gating Weights", fontsize=21)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/branch1_weights_{mse_diff}_{test_rs}_{test_noise}.pdf')

    # Visualize Branch 2 gating weights for Expert 1
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=branch2_weights[:, 0], cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Branch 2 - Expert 1 Weight", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("$x_0$", fontsize=25)
    plt.ylabel("$x_1$", fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.title("HMoE: Branch 2 Gating Weights", fontsize=21)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/branch2_weights_{mse_diff}_{test_rs}_{test_noise}.pdf')

    # Optionally, compare predictions vs true targets
    plt.figure(figsize=(8, 6))
    plt.scatter(true_vals.numpy(), preds_one.numpy(), alpha=0.6, label='MoE')
    plt.scatter(true_vals.numpy(), preds_two.numpy(), alpha=0.6, label='HMoE')
    plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'k--')
    plt.xlabel("True y", fontsize=25)
    plt.ylabel("Predicted y", fontsize=25)
    plt.xticks(fontsize=14)        
    plt.yticks(fontsize=14)
    plt.title("Comparison of Predictions", fontsize=27)
    plt.legend(fontsize=15)
    plt.savefig(f'/cis/home/xhan56/code/Multimodal-Transformer/src/figs/pred_{mse_diff}_{test_rs}_{test_noise}.pdf')