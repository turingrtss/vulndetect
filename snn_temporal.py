#!/usr/bin/env python3
"""
Where SNNs Beat CNNs — Testing on Temporal Tasks

The MNIST comparison showed SNNs match CNNs on static images but are 36x slower.
But MNIST is the worst case for temporal architectures. This experiment tests
tasks where SNN's temporal dynamics should be a natural advantage:

1. Sequential MNIST (pixels fed one at a time — pure temporal)
2. Temporal pattern detection (classify spike train patterns)
3. Event-based motion detection (moving vs stationary objects)

Hypothesis: SNNs will outperform CNNs on tasks with inherent temporal structure,
even on conventional CPU hardware.
"""

import time
import json
import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psutil

# Import SNN components from previous experiment
from snn_experiment import LIFLayer, SurrogateSpike, spike_fn


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ==================== TASK 1: Sequential MNIST ====================
# Feed pixels one at a time (784 timesteps, 1 pixel per step)
# This is a HARD sequence task — the model must remember early pixels

def load_mnist():
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return torch.FloatTensor(data.copy()) / 255.0
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            _, num = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.LongTensor(data.copy())
    base = os.path.expanduser('~/.cache/mnist/')
    return (read_images(base + 'train-images-idx3-ubyte.gz'),
            read_labels(base + 'train-labels-idx1-ubyte.gz'),
            read_images(base + 't10k-images-idx3-ubyte.gz'),
            read_labels(base + 't10k-labels-idx1-ubyte.gz'))


class SNN_Sequential(nn.Module):
    """SNN for sequential pixel-by-pixel input."""
    def __init__(self, input_size=1, hidden=128, num_classes=10, beta=0.95):
        super().__init__()
        self.lif1 = LIFLayer(input_size, hidden, beta=beta)
        self.lif2 = LIFLayer(hidden, num_classes, beta=beta)

    def forward(self, x):
        # x: (batch, seq_len) — one pixel per timestep
        batch, seq_len = x.shape
        mem1, mem2 = None, None
        spike_count = torch.zeros(batch, 10, device=x.device)

        for t in range(seq_len):
            inp = x[:, t:t+1]  # (batch, 1)
            spk1, mem1 = self.lif1(inp, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spike_count += spk2

        return spike_count / seq_len


class LSTM_Sequential(nn.Module):
    """LSTM for sequential input."""
    def __init__(self, input_size=1, hidden=128, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNN_Sequential(nn.Module):
    """1D CNN for sequential input."""
    def __init__(self, seq_len=784, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        return self.fc(self.conv(x))


class Transformer_Sequential(nn.Module):
    """Transformer for sequential input. Chunked to keep attention tractable."""
    def __init__(self, chunk_size=28, d_model=32, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(chunk_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 28, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 784) -> chunk into (batch, 28, 28)
        x = x.view(x.size(0), 28, 28)
        x = self.embed(x) + self.pos
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


# ==================== TASK 2: Temporal Pattern Classification ====================
# Classify spike train patterns — purely temporal, no spatial structure

def generate_temporal_patterns(n_samples=5000, seq_len=200, n_classes=5):
    """Generate synthetic temporal spike patterns for classification."""
    X = torch.zeros(n_samples, seq_len)
    y = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        label = i % n_classes
        y[i] = label

        if label == 0:  # Regular bursts
            period = 20
            for t in range(0, seq_len, period):
                burst_len = 5
                X[i, t:min(t+burst_len, seq_len)] = 1.0

        elif label == 1:  # Accelerating frequency
            freq = 2.0
            for t in range(seq_len):
                freq += 0.05
                if np.random.random() < min(freq / 100, 0.8):
                    X[i, t] = 1.0

        elif label == 2:  # Decelerating frequency
            freq = 80.0
            for t in range(seq_len):
                freq -= 0.3
                if np.random.random() < max(freq / 100, 0.02):
                    X[i, t] = 1.0

        elif label == 3:  # Two-rhythm pattern (fast then slow)
            for t in range(seq_len):
                if t < seq_len // 2:
                    if np.random.random() < 0.6:
                        X[i, t] = 1.0
                else:
                    if np.random.random() < 0.1:
                        X[i, t] = 1.0

        elif label == 4:  # Random with specific inter-spike intervals
            t = 0
            while t < seq_len:
                X[i, t] = 1.0
                t += np.random.choice([3, 3, 3, 15, 15])  # Bimodal ISI

    # Add noise
    noise = (torch.rand_like(X) < 0.05).float()
    X = torch.clamp(X + noise, 0, 1)

    return X, y


class SNN_Temporal(nn.Module):
    """SNN for temporal pattern classification."""
    def __init__(self, hidden=64, num_classes=5, beta=0.9):
        super().__init__()
        self.lif1 = LIFLayer(1, hidden, beta=beta)
        self.lif2 = LIFLayer(hidden, num_classes, beta=beta)

    def forward(self, x):
        batch, seq_len = x.shape
        mem1, mem2 = None, None
        spike_count = torch.zeros(batch, 5, device=x.device)

        for t in range(seq_len):
            inp = x[:, t:t+1]
            spk1, mem1 = self.lif1(inp, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spike_count += spk2

        return spike_count / seq_len


class LSTM_Temporal(nn.Module):
    def __init__(self, hidden=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNN_Temporal(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, num_classes))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


# ==================== TRAINING ====================

def train_eval(model, name, X_train, y_train, X_test, y_test, epochs=15, batch_size=128, lr=1e-3):
    print(f"\n{'='*50}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"{'='*50}")

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: acc={correct/total:.4f}")
    train_time = time.time() - start

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), batch_size):
            preds.append(model(X_test[i:i+batch_size]).argmax(1))
        y_pred = torch.cat(preds)
    infer_time = time.time() - t0

    acc = (y_pred == y_test).float().mean().item()
    print(f"  Test Acc: {acc:.4f} | Train: {train_time:.1f}s | Infer: {infer_time/len(X_test)*1000:.4f}ms")

    return {
        'name': name, 'params': params, 'test_acc': round(acc, 4),
        'train_time_s': round(train_time, 1),
        'infer_ms': round(infer_time / len(X_test) * 1000, 4),
    }


def main():
    print("="*60)
    print("Where SNNs Beat CNNs — Temporal Task Comparison")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print("="*60)

    all_results = {}

    # === TASK 1: Sequential MNIST (pixel-by-pixel, 784 steps) ===
    print("\n\n" + "#"*60)
    print("TASK 1: Sequential MNIST (784 timesteps, 1 pixel/step)")
    print("#"*60)

    X_train, y_train, X_test, y_test = load_mnist()
    # Flatten to sequences and subsample
    X_tr = X_train[:10000].view(10000, -1)  # (10000, 784)
    y_tr = y_train[:10000]
    X_te = X_test[:2000].view(2000, -1)
    y_te = y_test[:2000]

    # Use smaller seq for feasibility: chunk to 196 steps of 4 pixels
    # Actually let's use row-by-row (28 steps of 28) for speed
    print("  (Using row-by-row: 28 timesteps of 28 pixels for feasibility)")
    X_tr_seq = X_tr.view(10000, 28, 28)
    X_te_seq = X_te.view(2000, 28, 28)

    # SNN processes rows as timesteps
    class SNN_RowSeq(nn.Module):
        def __init__(self, input_size=28, hidden=64, num_classes=10, beta=0.9):
            super().__init__()
            self.lif1 = LIFLayer(input_size, hidden, beta=beta)
            self.lif2 = LIFLayer(hidden, num_classes, beta=beta)
        def forward(self, x):
            batch, steps, features = x.shape
            mem1, mem2 = None, None
            out = torch.zeros(batch, 10, device=x.device)
            for t in range(steps):
                spk1, mem1 = self.lif1(x[:, t, :], mem1)
                spk2, mem2 = self.lif2(spk1, mem2)
                out += spk2
            return out / steps

    class LSTM_RowSeq(nn.Module):
        def __init__(self, input_size=28, hidden=64, num_classes=10):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class CNN_1D_RowSeq(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(28, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            )
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*4, num_classes))
        def forward(self, x):
            # x: (batch, 28, 28) -> treat rows as channels
            return self.fc(self.conv(x))

    all_results['seq_snn'] = train_eval(SNN_RowSeq(), "SNN (row-seq)", X_tr_seq, y_tr, X_te_seq, y_te)
    all_results['seq_lstm'] = train_eval(LSTM_RowSeq(), "LSTM (row-seq)", X_tr_seq, y_tr, X_te_seq, y_te)
    all_results['seq_cnn'] = train_eval(CNN_1D_RowSeq(), "CNN-1D (row-seq)", X_tr_seq, y_tr, X_te_seq, y_te)

    # === TASK 2: Temporal Spike Pattern Classification ===
    print("\n\n" + "#"*60)
    print("TASK 2: Temporal Spike Pattern Classification")
    print("#"*60)

    X_spk, y_spk = generate_temporal_patterns(n_samples=8000, seq_len=200, n_classes=5)
    X_tr2, X_te2 = X_spk[:6000], X_spk[6000:]
    y_tr2, y_te2 = y_spk[:6000], y_spk[6000:]

    all_results['temp_snn'] = train_eval(SNN_Temporal(64, 5), "SNN (temporal)", X_tr2, y_tr2, X_te2, y_te2)
    all_results['temp_lstm'] = train_eval(LSTM_Temporal(64, 5), "LSTM (temporal)", X_tr2, y_tr2, X_te2, y_te2)
    all_results['temp_cnn'] = train_eval(CNN_Temporal(5), "CNN-1D (temporal)", X_tr2, y_tr2, X_te2, y_te2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Where Do SNNs Win?")
    print("="*60)

    print(f"\n{'Task':<12} {'Architecture':<20} {'Params':>8} {'Acc':>8} {'Train(s)':>9} {'Infer(ms)':>10}")
    print("-"*67)

    for task_prefix, task_name in [('seq_', 'SeqMNIST'), ('temp_', 'Temporal')]:
        task_results = [(k, v) for k, v in all_results.items() if k.startswith(task_prefix)]
        for k, r in sorted(task_results, key=lambda x: -x[1]['test_acc']):
            print(f"{task_name:<12} {r['name']:<20} {r['params']:>8,} {r['test_acc']:>8.4f} {r['train_time_s']:>9.1f} {r['infer_ms']:>10.4f}")
        print()

    with open('snn_temporal_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to snn_temporal_results.json")


if __name__ == '__main__':
    main()
