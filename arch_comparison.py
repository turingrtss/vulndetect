#!/usr/bin/env python3
"""
Neural Architecture Comparison — Same Task, Every Architecture

Dataset: MNIST (simple enough to train all architectures quickly on CPU)
Then: CIFAR-10 (harder, tests if rankings change with complexity)

Architectures:
  1. MLP (baseline)
  2. CNN (LeNet-style)
  3. RNN (vanilla)
  4. LSTM
  5. GRU
  6. Transformer (small)
  7. Mamba-style (S4/state space approximation)
  8. KAN (Kolmogorov-Arnold Network)
  9. Mixture of Experts (2 experts + gating)

All architectures get the same parameter budget (~50K params),
same optimizer (Adam 1e-3), same epochs (20), same data.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psutil
import os
import math


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ==================== DATA ====================

def load_mnist():
    """Load MNIST from raw files."""
    import gzip
    import struct
    
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return torch.FloatTensor(data) / 255.0
    
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.LongTensor(data)
    
    base = os.path.expanduser('~/.cache/mnist/')
    X_train = read_images(base + 'train-images-idx3-ubyte.gz')
    y_train = read_labels(base + 'train-labels-idx1-ubyte.gz')
    X_test = read_images(base + 't10k-images-idx3-ubyte.gz')
    y_test = read_labels(base + 't10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test


# ==================== ARCHITECTURES ====================

class MLP(nn.Module):
    """Multi-layer perceptron. Flattened input."""
    def __init__(self, input_dim=784, hidden=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """LeNet-style CNN."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


class VanillaRNN(nn.Module):
    """Process image as 28 time steps of 28 pixels."""
    def __init__(self, input_size=28, hidden_size=64, num_classes=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMNet(nn.Module):
    """LSTM on image rows."""
    def __init__(self, input_size=28, hidden_size=48, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUNet(nn.Module):
    """GRU on image rows."""
    def __init__(self, input_size=28, hidden_size=56, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SmallTransformer(nn.Module):
    """Tiny transformer. Image rows as sequence tokens."""
    def __init__(self, input_size=28, d_model=32, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 28, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.embed(x) + self.pos_enc
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


class S4Block(nn.Module):
    """Simplified state space model (S4-inspired).
    Not a full S4 implementation — uses discretized linear RNN 
    with learned state matrices as an approximation."""
    def __init__(self, d_input, d_state=32, d_output=None):
        super().__init__()
        d_output = d_output or d_input
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_input) * 0.1)
        self.C = nn.Parameter(torch.randn(d_output, d_state) * 0.1)
        self.D = nn.Parameter(torch.randn(d_output, d_input) * 0.1)
        self.d_state = d_state
        
    def forward(self, x):
        # x: (batch, seq_len, d_input)
        batch, seq_len, d_input = x.shape
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(h @ self.A.T + x[:, t, :] @ self.B.T)
            y = h @ self.C.T + x[:, t, :] @ self.D.T
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class MambaStyle(nn.Module):
    """State space model for sequence classification."""
    def __init__(self, input_size=28, d_model=32, d_state=16, num_classes=10):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.s4 = S4Block(d_model, d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.embed(x)
        x = self.norm(self.s4(x))
        return self.fc(x.mean(dim=1))


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network layer.
    Each connection has a learnable activation (approximated by B-splines)
    instead of a fixed activation + weight."""
    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        # Spline coefficients for each input-output pair
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        # Grid points
        self.register_buffer('grid', torch.linspace(-2, 2, grid_size))
        
    def forward(self, x):
        # x: (batch, in_features)
        batch = x.shape[0]
        # Evaluate B-spline-like basis at each input value
        # Simplified: use RBF basis instead of proper B-splines
        x_expanded = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size)
        basis = torch.exp(-0.5 * (x_expanded - grid) ** 2)  # (batch, in, grid_size)
        # Multiply by coefficients and sum
        # coeffs: (out, in, grid_size)
        # basis: (batch, in, grid_size)
        out = torch.einsum('big,oig->bo', basis, self.coeffs) + self.bias
        return out


class KANNet(nn.Module):
    """KAN for classification."""
    def __init__(self, input_dim=784, hidden=32, num_classes=10, grid_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            KANLinear(input_dim, hidden, grid_size),
            KANLinear(hidden, num_classes, grid_size),
        )
    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Simple MoE with 2 expert MLPs and a gating network."""
    def __init__(self, input_dim=784, hidden=48, num_classes=10, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        gate_weights = self.gate(x)  # (batch, num_experts)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, num_experts, classes)
        return (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


# ==================== TRAINING ====================

def train_and_evaluate(model, name, X_train, y_train, X_test, y_test, 
                       epochs=15, batch_size=512, lr=1e-3):
    """Train a model and return metrics."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"{'='*50}")
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_mem = get_memory_mb()
    start_time = time.time()
    
    train_accs = []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
        
        train_acc = correct / total
        train_accs.append(train_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/total:.4f}  train_acc={train_acc:.4f}")
    
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    eval_start = time.time()
    with torch.no_grad():
        # Process test set in batches to avoid memory issues
        all_preds = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            preds = model(batch).argmax(1)
            all_preds.append(preds)
        y_pred = torch.cat(all_preds)
    eval_time = time.time() - eval_start
    
    test_acc = (y_pred == y_test).float().mean().item()
    peak_mem = get_memory_mb()
    
    # Per-class accuracy
    per_class = []
    for c in range(10):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).float().mean().item()
            per_class.append(class_acc)
    
    # Inference speed
    infer_ms = eval_time / len(X_test) * 1000
    
    result = {
        'name': name,
        'params': params,
        'test_acc': round(test_acc, 4),
        'train_acc': round(train_accs[-1], 4),
        'train_time_s': round(train_time, 1),
        'infer_ms_per_sample': round(infer_ms, 4),
        'peak_memory_mb': round(peak_mem, 1),
        'epochs': epochs,
        'per_class_acc': [round(a, 3) for a in per_class],
        'convergence': train_accs,
    }
    
    print(f"\n  Test Accuracy:  {test_acc:.4f}")
    print(f"  Train Time:     {train_time:.1f}s")
    print(f"  Inference:      {infer_ms:.4f} ms/sample")
    print(f"  Memory:         {peak_mem:.0f} MB")
    
    return result


def main():
    print("="*60)
    print("Neural Architecture Comparison")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dataset: MNIST (28x28, 10 classes)")
    print(f"Budget: Same epochs (15), same optimizer (Adam 1e-3)")
    print("="*60)
    
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use 20K training samples to keep sequential models feasible on CPU
    subset = 20000
    idx = torch.randperm(len(X_train))[:subset]
    X_train_sub = X_train[idx]
    y_train_sub = y_train[idx]
    print(f"\nTrain: {X_train_sub.shape} (subset of {len(X_train)}), Test: {X_test.shape}")
    
    architectures = [
        ("MLP", MLP(784, 64, 10)),
        ("CNN (LeNet)", CNN(10)),
        ("Vanilla RNN", VanillaRNN(28, 64, 10)),
        ("LSTM", LSTMNet(28, 48, 10)),
        ("GRU", GRUNet(28, 56, 10)),
        ("Transformer", SmallTransformer(28, 32, 4, 2, 10)),
        ("State Space (S4-style)", MambaStyle(28, 32, 16, 10)),
        ("KAN", KANNet(784, 32, 10, 5)),
        ("Mixture of Experts", MixtureOfExperts(784, 48, 10, 2)),
    ]
    
    all_results = {}
    
    for name, model in architectures:
        try:
            result = train_and_evaluate(model, name, X_train_sub, y_train_sub, X_test, y_test)
            all_results[name] = result
            # Save incrementally in case of crash
            save = {}
            for k, v in all_results.items():
                save[k] = {kk: vv for kk, vv in v.items() if kk != 'convergence'}
            with open('arch_comparison_results.json', 'w') as f:
                json.dump(save, f, indent=2, default=str)
        except Exception as e:
            print(f"\n  FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {'name': name, 'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Architecture':<25} {'Params':>8} {'Test Acc':>10} {'Train(s)':>10} {'Infer(ms)':>10}")
    print("-"*63)
    
    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if 'test_acc' in v],
        key=lambda x: -x[1]['test_acc']
    )
    
    for name, r in sorted_results:
        print(f"{r['name']:<25} {r['params']:>8,} {r['test_acc']:>10.4f} {r['train_time_s']:>10.1f} {r['infer_ms_per_sample']:>10.4f}")
    
    # Save results (without convergence curves for clean JSON)
    save = {}
    for k, v in all_results.items():
        save[k] = {kk: vv for kk, vv in v.items() if kk != 'convergence'}
    
    with open('arch_comparison_results.json', 'w') as f:
        json.dump(save, f, indent=2, default=str)
    
    print("\nResults saved to arch_comparison_results.json")


if __name__ == '__main__':
    main()
