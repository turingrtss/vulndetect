#!/usr/bin/env python3
"""
Spiking Neural Networks on CPU
Experiment #17: Neuromorphic computing without neuromorphic hardware

Implements leaky integrate-and-fire (LIF) neurons with surrogate gradient
training. Compares against the conventional architectures from the previous
experiment on the same MNIST task.

Key questions:
  1. Can SNNs match conventional networks on accuracy?
  2. What's the FLOPs/inference difference?
  3. Is there any practical advantage on conventional CPU hardware?
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


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


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


# ==================== SURROGATE GRADIENT ====================

class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for the Heaviside step function.
    Forward: spike = 1 if membrane > threshold, else 0
    Backward: use a smooth approximation (fast sigmoid)."""
    scale = 25.0

    @staticmethod
    def forward(ctx, membrane):
        ctx.save_for_backward(membrane)
        return (membrane > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        grad = grad_output / (SurrogateSpike.scale * torch.abs(membrane) + 1.0) ** 2
        return grad


spike_fn = SurrogateSpike.apply


# ==================== LIF NEURON LAYER ====================

class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.
    
    Dynamics per timestep:
      membrane = beta * membrane_prev + input
      spike = Heaviside(membrane - threshold)
      membrane = membrane * (1 - spike)  # reset after spike
    """
    def __init__(self, in_features, out_features, beta=0.9, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.beta = beta          # Membrane decay constant
        self.threshold = threshold

    def forward(self, x, mem=None):
        """x: (batch, in_features). Returns (spikes, new_membrane)."""
        batch = x.shape[0]
        if mem is None:
            mem = torch.zeros(batch, self.linear.out_features, device=x.device)

        cur = self.linear(x)
        mem = self.beta * mem + cur
        spikes = spike_fn(mem - self.threshold)
        mem = mem * (1 - spikes)  # Reset
        return spikes, mem


class LIFConvLayer(nn.Module):
    """Convolutional LIF layer."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 beta=0.9, threshold=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(self.conv(x))

        cur = self.conv(x)
        mem = self.beta * mem + cur
        spikes = spike_fn(mem - self.threshold)
        mem = mem * (1 - spikes)
        return spikes, mem


# ==================== SNN MODELS ====================

class SNN_FC(nn.Module):
    """Fully connected SNN. Processes input over T timesteps."""
    def __init__(self, input_dim=784, hidden=128, num_classes=10,
                 timesteps=25, beta=0.9):
        super().__init__()
        self.timesteps = timesteps
        self.lif1 = LIFLayer(input_dim, hidden, beta=beta)
        self.lif2 = LIFLayer(hidden, num_classes, beta=beta)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten

        mem1 = None
        mem2 = None
        spike_count = torch.zeros(x.size(0), 10, device=x.device)

        for t in range(self.timesteps):
            # Rate coding: input spikes with probability = pixel intensity
            inp = (torch.rand_like(x) < x).float()

            spk1, mem1 = self.lif1(inp, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spike_count += spk2

        return spike_count / self.timesteps


class SNN_Conv(nn.Module):
    """Convolutional SNN."""
    def __init__(self, num_classes=10, timesteps=25, beta=0.9):
        super().__init__()
        self.timesteps = timesteps
        self.lif_conv1 = LIFConvLayer(1, 8, 5, padding=2, beta=beta)
        self.pool1 = nn.MaxPool2d(2)
        self.lif_conv2 = LIFConvLayer(8, 16, 5, padding=2, beta=beta)
        self.pool2 = nn.MaxPool2d(2)
        self.lif_fc = LIFLayer(16 * 7 * 7, num_classes, beta=beta)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        mem1 = None
        mem2 = None
        mem_fc = None
        spike_count = torch.zeros(x.size(0), 10, device=x.device)

        for t in range(self.timesteps):
            inp = (torch.rand(x.shape, device=x.device) < x).float()

            spk1, mem1 = self.lif_conv1(inp, mem1)
            spk1 = self.pool1(spk1)
            spk2, mem2 = self.lif_conv2(spk1, mem2)
            spk2 = self.pool2(spk2)
            spk2_flat = spk2.view(spk2.size(0), -1)
            spk_out, mem_fc = self.lif_fc(spk2_flat, mem_fc)
            spike_count += spk_out

        return spike_count / self.timesteps


# ==================== TRAINING ====================

def count_flops_per_inference(model, input_shape, timesteps):
    """Estimate FLOPs per inference (multiply-accumulate operations)."""
    flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            flops += module.in_features * module.out_features * 2  # multiply + add
        elif isinstance(module, nn.Conv2d):
            # Approximate: out_channels * out_h * out_w * kernel_h * kernel_w * in_channels * 2
            # Simplified estimate
            flops += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * 100
    return flops * timesteps


def train_model(model, name, X_train, y_train, X_test, y_test,
                epochs=15, batch_size=256, lr=1e-3):
    print(f"\n{'='*50}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    if hasattr(model, 'timesteps'):
        print(f"  Timesteps: {model.timesteps}")
    print(f"{'='*50}")

    train_ds = TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)

        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/total:.4f}  acc={acc:.4f}")

    train_time = time.time() - start

    # Evaluate
    model.eval()
    eval_start = time.time()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(X_test), batch_size):
            preds = model(X_test[i:i+batch_size]).argmax(1)
            all_preds.append(preds)
        y_pred = torch.cat(all_preds)
    eval_time = time.time() - eval_start

    test_acc = (y_pred == y_test).float().mean().item()
    infer_ms = eval_time / len(X_test) * 1000

    # Count spike operations (proxy for energy)
    total_spikes = 0
    if hasattr(model, 'timesteps'):
        # Run one batch and count spikes
        model.eval()
        with torch.no_grad():
            sample = X_test[:100]
            # Hook to count spikes
            spike_counts = []
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    spikes = output[0]
                    spike_counts.append(spikes.sum().item())
            hooks = []
            for m in model.modules():
                if isinstance(m, (LIFLayer, LIFConvLayer)):
                    hooks.append(m.register_forward_hook(hook_fn))
            model(sample)
            for h in hooks:
                h.remove()
            total_spikes = sum(spike_counts) / 100  # per sample

    result = {
        'name': name,
        'params': params,
        'test_acc': round(test_acc, 4),
        'train_time_s': round(train_time, 1),
        'infer_ms': round(infer_ms, 4),
        'memory_mb': round(get_memory_mb(), 1),
        'avg_spikes_per_sample': round(total_spikes, 1),
        'timesteps': getattr(model, 'timesteps', 0),
    }

    print(f"\n  Test Accuracy:  {test_acc:.4f}")
    print(f"  Train Time:     {train_time:.1f}s")
    print(f"  Inference:      {infer_ms:.4f} ms/sample")
    if total_spikes > 0:
        print(f"  Avg Spikes:     {total_spikes:.1f} per sample")

    return result


def main():
    print("="*60)
    print("Spiking Neural Networks on CPU")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print(f"PyTorch: {torch.__version__}")
    print("="*60)

    X_train, y_train, X_test, y_test = load_mnist()
    subset = 20000
    idx = torch.randperm(len(X_train))[:subset]
    X_train = X_train[idx]
    y_train = y_train[idx]
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    # SNN variants with different timesteps
    for T in [10, 25, 50]:
        name = f"SNN-FC (T={T})"
        model = SNN_FC(784, 128, 10, timesteps=T, beta=0.9)
        results[name] = train_model(model, name, X_train, y_train, X_test, y_test)

    # Convolutional SNN
    for T in [10, 25]:
        name = f"SNN-Conv (T={T})"
        model = SNN_Conv(10, timesteps=T, beta=0.9)
        results[name] = train_model(model, name, X_train, y_train, X_test, y_test)

    # Conventional baselines for direct comparison
    from arch_comparison import MLP, CNN, SmallTransformer

    results['MLP (conventional)'] = train_model(
        MLP(784, 64, 10), 'MLP (conventional)',
        X_train, y_train, X_test, y_test)

    results['CNN (conventional)'] = train_model(
        CNN(10), 'CNN (conventional)',
        X_train, y_train, X_test, y_test)

    results['Transformer (conventional)'] = train_model(
        SmallTransformer(28, 32, 4, 2, 10), 'Transformer (conventional)',
        X_train, y_train, X_test, y_test)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Spiking vs Conventional")
    print("="*60)

    print(f"\n{'Architecture':<30} {'Params':>8} {'Acc':>8} {'Train(s)':>9} {'Infer(ms)':>10} {'Spikes':>8}")
    print("-"*73)

    for name, r in sorted(results.items(), key=lambda x: -x[1].get('test_acc', 0)):
        spk = f"{r['avg_spikes_per_sample']:.0f}" if r.get('avg_spikes_per_sample', 0) > 0 else "-"
        print(f"{r['name']:<30} {r['params']:>8,} {r['test_acc']:>8.4f} {r['train_time_s']:>9.1f} {r['infer_ms']:>10.4f} {spk:>8}")

    with open('snn_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to snn_results.json")


if __name__ == '__main__':
    main()
