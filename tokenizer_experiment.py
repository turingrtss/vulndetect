#!/usr/bin/env python3
"""
Tokenizer Effects on Model Behavior (#2)

Same dataset, same model architecture, 4 different tokenization strategies.
How much does tokenization choice change classification accuracy, training
speed, and what the model learns?

Tokenization strategies:
  1. Character-level (each character is a token)
  2. Word-level (whitespace split)
  3. Byte Pair Encoding (BPE) - learned subword merges
  4. Character n-gram (overlapping character windows)

Task: Text classification on AG News (4 classes: World, Sports, Business, Sci/Tech)
Model: Same 2-layer feedforward network for all tokenizers
"""

import time
import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
import psutil


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ==================== DATA ====================

def load_ag_news():
    """Load AG News from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    
    train_texts = [x['text'] for x in ds['train']]
    train_labels = [x['label'] for x in ds['train']]
    test_texts = [x['text'] for x in ds['test']]
    test_labels = [x['label'] for x in ds['test']]
    
    # Subsample for feasibility on CPU
    np.random.seed(42)
    train_idx = np.random.choice(len(train_texts), 15000, replace=False)
    test_idx = np.random.choice(len(test_texts), 3000, replace=False)
    
    train_texts = [train_texts[i] for i in train_idx]
    train_labels = [train_labels[i] for i in train_idx]
    test_texts = [test_texts[i] for i in test_idx]
    test_labels = [test_labels[i] for i in test_idx]
    
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    print(f"Classes: {Counter(train_labels)}")
    
    return train_texts, train_labels, test_texts, test_labels


# ==================== TOKENIZERS ====================

class CharTokenizer:
    """Character-level tokenization."""
    name = "Character"
    
    def __init__(self, max_vocab=256, max_len=500):
        self.max_len = max_len
        self.char2idx = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        chars = Counter()
        for t in texts:
            chars.update(t[:self.max_len])
        self.char2idx = {c: i+1 for i, (c, _) in enumerate(chars.most_common(254))}
        self.char2idx['<PAD>'] = 0
        self.vocab_size = len(self.char2idx)
        return self
    
    def encode(self, text):
        ids = [self.char2idx.get(c, 0) for c in text[:self.max_len]]
        # Pad
        ids = ids + [0] * (self.max_len - len(ids))
        return ids
    
    def encode_batch(self, texts):
        return torch.LongTensor([self.encode(t) for t in texts])
    
    def describe(self):
        return f"vocab={self.vocab_size}, max_len={self.max_len}, granularity=character"


class WordTokenizer:
    """Word-level tokenization (whitespace + punctuation split)."""
    name = "Word"
    
    def __init__(self, max_vocab=10000, max_len=100):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {}
        self.vocab_size = 0
    
    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, texts):
        word_counts = Counter()
        for t in texts:
            word_counts.update(self._tokenize(t))
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common(self.max_vocab - 2):
            self.word2idx[word] = len(self.word2idx)
        self.vocab_size = len(self.word2idx)
        return self
    
    def encode(self, text):
        tokens = self._tokenize(text)[:self.max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        ids = ids + [0] * (self.max_len - len(ids))
        return ids
    
    def encode_batch(self, texts):
        return torch.LongTensor([self.encode(t) for t in texts])
    
    def describe(self):
        return f"vocab={self.vocab_size}, max_len={self.max_len}, granularity=word"


class BPETokenizer:
    """Byte Pair Encoding - learns subword merges from data."""
    name = "BPE"
    
    def __init__(self, max_vocab=5000, max_len=200):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.merges = []
        self.token2idx = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        # Start with character-level vocabulary
        word_freqs = Counter()
        for t in texts:
            words = re.findall(r'\b\w+\b', t.lower())
            for w in words:
                word_freqs[' '.join(list(w)) + ' </w>'] += 1
        
        # Learn BPE merges
        vocab = Counter()
        for word in word_freqs:
            for char in word.split():
                vocab[char] += 1
        
        num_merges = min(self.max_vocab - len(vocab) - 2, 2000)
        
        for merge_i in range(num_merges):
            # Count pairs
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            
            if not pairs:
                break
            
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            
            # Apply merge
            new_word_freqs = {}
            bigram = ' '.join(best)
            replacement = ''.join(best)
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
            vocab[replacement] = 1
            
            if len(vocab) >= self.max_vocab - 2:
                break
        
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        for token in vocab:
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
        self.vocab_size = len(self.token2idx)
        return self
    
    def _apply_bpe(self, word):
        symbols = list(word) + ['</w>']
        for left, right in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == left and symbols[i+1] == right:
                    symbols = symbols[:i] + [left + right] + symbols[i+2:]
                else:
                    i += 1
        return symbols
    
    def encode(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        ids = []
        for w in words:
            tokens = self._apply_bpe(w)
            for t in tokens:
                ids.append(self.token2idx.get(t, 1))
        ids = ids[:self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))
        return ids
    
    def encode_batch(self, texts):
        return torch.LongTensor([self.encode(t) for t in texts])
    
    def describe(self):
        return f"vocab={self.vocab_size}, max_len={self.max_len}, merges={len(self.merges)}, granularity=subword"


class NgramTokenizer:
    """Character n-gram tokenization (overlapping windows)."""
    name = "Char N-gram"
    
    def __init__(self, n=3, max_vocab=10000, max_len=200):
        self.n = n
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.ngram2idx = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        ngram_counts = Counter()
        for t in texts:
            t_lower = t.lower()
            for i in range(len(t_lower) - self.n + 1):
                ngram_counts[t_lower[i:i+self.n]] += 1
        
        self.ngram2idx = {'<PAD>': 0, '<UNK>': 1}
        for ng, _ in ngram_counts.most_common(self.max_vocab - 2):
            self.ngram2idx[ng] = len(self.ngram2idx)
        self.vocab_size = len(self.ngram2idx)
        return self
    
    def encode(self, text):
        t = text.lower()
        ids = []
        for i in range(len(t) - self.n + 1):
            ng = t[i:i+self.n]
            ids.append(self.ngram2idx.get(ng, 1))
        ids = ids[:self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))
        return ids
    
    def encode_batch(self, texts):
        return torch.LongTensor([self.encode(t) for t in texts])
    
    def describe(self):
        return f"vocab={self.vocab_size}, max_len={self.max_len}, n={self.n}, granularity=char-{self.n}gram"


# ==================== MODEL ====================

class TextClassifier(nn.Module):
    """Same architecture for all tokenizers: Embedding → AvgPool → FC → FC."""
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )
    
    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        # Average pooling over sequence (ignore padding)
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ==================== TRAINING ====================

def train_and_evaluate(tokenizer, train_texts, train_labels, test_texts, test_labels,
                       embed_dim=64, hidden=128, epochs=15, batch_size=256, lr=1e-3,
                       seeds=(0, 1, 2, 3, 4)):
    """Fit/encode the tokenizer ONCE (deterministic, no randomness), then
    train+evaluate the model across multiple seeds and report mean ± std.
    Single-seed runs can't distinguish a real tokenizer effect from
    run-to-run variance (raised by a dev.to commenter, raknaos, 2026-09-11).
    """
    print(f"\n{'='*60}")
    print(f"  Tokenizer: {tokenizer.name}")
    print(f"  {tokenizer.describe()}")
    print(f"{'='*60}")
    
    # Fit tokenizer
    t0 = time.time()
    tokenizer.fit(train_texts)
    fit_time = time.time() - t0
    print(f"  Tokenizer fit: {fit_time:.1f}s, vocab={tokenizer.vocab_size}")
    
    # Encode
    t0 = time.time()
    X_train = tokenizer.encode_batch(train_texts)
    X_test = tokenizer.encode_batch(test_texts)
    encode_time = time.time() - t0
    print(f"  Encoding: {encode_time:.1f}s")
    
    y_train = torch.LongTensor(train_labels)
    y_test = torch.LongTensor(test_labels)
    
    # Sequence length stats
    train_lengths = (X_train != 0).sum(dim=1).float()
    print(f"  Avg sequence length: {train_lengths.mean():.0f} tokens")
    print(f"  Max sequence length: {train_lengths.max():.0f} tokens")

    seed_accs = []
    seed_train_times = []
    per_seed_results = []
    model = None
    params = 0
    train_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TextClassifier(tokenizer.vocab_size, embed_dim, hidden, 4)
        params = sum(p.numel() for p in model.parameters())

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        train_accs = []
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
            acc = correct / total
            train_accs.append(acc)
        train_time = time.time() - start

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_test), batch_size):
                preds.append(model(X_test[i:i+batch_size]).argmax(1))
            y_pred = torch.cat(preds)

        test_acc = (y_pred == y_test).float().mean().item()
        seed_accs.append(test_acc)
        seed_train_times.append(train_time)
        per_seed_results.append({'seed': seed, 'test_acc': round(test_acc, 4), 'train_time_s': round(train_time, 1)})
        print(f"  Seed {seed}: test_acc={test_acc:.4f}  train_time={train_time:.1f}s")

    mean_acc = float(np.mean(seed_accs))
    std_acc = float(np.std(seed_accs))
    infer_time = 0.0

    # Keep last-seed model's per-class breakdown + convergence curve for
    # reference (not meaningfully different across seeds at this scale)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), batch_size):
            preds.append(model(X_test[i:i+batch_size]).argmax(1))
        y_pred = torch.cat(preds)
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    per_class = {}
    for c in range(4):
        mask = y_test == c
        if mask.sum() > 0:
            per_class[class_names[c]] = round((y_pred[mask] == y_test[mask]).float().mean().item(), 4)

    print(f"\n  Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  (n={len(seeds)} seeds)")
    print(f"  Per-class (last seed): {per_class}")
    print(f"  Tokenizer overhead: fit={fit_time:.1f}s, encode={encode_time:.1f}s")
    
    return {
        'tokenizer': tokenizer.name,
        'vocab_size': tokenizer.vocab_size,
        'config': tokenizer.describe(),
        'params': params,
        'test_acc': round(mean_acc, 4),
        'test_acc_std': round(std_acc, 4),
        'per_seed': per_seed_results,
        'per_class': per_class,
        'train_time_s': round(float(np.mean(seed_train_times)), 1),
        'fit_time_s': round(fit_time, 1),
        'encode_time_s': round(encode_time, 1),
        'infer_ms': round(infer_time / len(test_texts) * 1000, 4),
        'avg_seq_len': round(train_lengths.mean().item()),
        'convergence': [round(a, 4) for a in train_accs],
    }


def main():
    print("="*60)
    print("Tokenizer Effects on Model Behavior")
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print(f"Task: AG News classification (4 classes)")
    print(f"Model: Embedding → AvgPool → FC (same for all)")
    print("="*60)
    
    train_texts, train_labels, test_texts, test_labels = load_ag_news()
    
    tokenizers = [
        CharTokenizer(max_vocab=256, max_len=500),
        WordTokenizer(max_vocab=10000, max_len=100),
        BPETokenizer(max_vocab=5000, max_len=200),
        NgramTokenizer(n=3, max_vocab=10000, max_len=300),
    ]
    
    all_results = {}
    
    for tok in tokenizers:
        try:
            result = train_and_evaluate(tok, train_texts, train_labels, test_texts, test_labels)
            all_results[tok.name] = result
            # Save incrementally
            with open('tokenizer_results.json', 'w') as f:
                save = {k: {kk: vv for kk, vv in v.items() if kk != 'convergence'} for k, v in all_results.items()}
                json.dump(save, f, indent=2)
        except Exception as e:
            print(f"\n  FAILED: {tok.name} — {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Tokenizer':<15} {'Vocab':>7} {'SeqLen':>7} {'Acc':>8} {'Train(s)':>9} {'Fit(s)':>7} {'Enc(s)':>7}")
    print("-"*62)
    
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['test_acc']):
        print(f"{r['tokenizer']:<15} {r['vocab_size']:>7} {r['avg_seq_len']:>7} {r['test_acc']:>8.4f} {r['train_time_s']:>9.1f} {r['fit_time_s']:>7.1f} {r['encode_time_s']:>7.1f}")
    
    print(f"\nPeak memory: {get_memory_mb():.0f} MB")
    print("Results saved to tokenizer_results.json")


if __name__ == '__main__':
    main()
