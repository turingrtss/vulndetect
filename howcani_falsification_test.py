#!/usr/bin/env python3
"""
Answering howcani's falsification test + ablation request (dev.to, 2026-09-12):

  1. PERMUTATION TEST: permute comment length across labels (shuffle which
     document gets which original-comment-length-derived filler, breaking
     the length<->label correlation while preserving the realistic length
     DISTRIBUTION) -- rerun the filler-arm experiment, check whether the
     " # " marker's coefficient collapses toward zero. If it does, that
     confirms the marker's predictive power comes from the length<->label
     correlation (a real leak), not from the marker token itself carrying
     independent signal.

  2. MARKER ABLATION: drop the " # " token from the vocabulary entirely,
     refit, report accuracy on the filler arm without it. This estimates
     how much of the filler arm's reported accuracy the marker alone
     was responsible for.

Reuses the same shared-vectorizer pipeline as leakage_followup.py so
results are directly comparable.
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/vulndetect')
import numpy as np
import json
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from leakage_followup import extract_comment_blocks, length_matched_filler

print("Loading dataset...")
ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
codes = [x['code'] for x in ds['train']]
labels = [x['label'] for x in ds['train']]
binary_labels = [0 if l == 'safe' else 1 for l in labels]

X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
    codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
train_idx, test_idx = train_test_split(
    list(range(len(codes))), test_size=0.2, random_state=42, stratify=binary_labels)

shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                     max_features=50000, sublinear_tf=True)
shared_vectorizer.fit(X_train_baseline)
feature_names = list(shared_vectorizer.get_feature_names_out())
marker = " # "
marker_idx = feature_names.index(marker) if marker in feature_names else None
print(f"Marker index: {marker_idx}")

# ---------- Original filler arm (baseline for comparison) ----------
codes_filler = length_matched_filler(codes, seed=42)
X_train_filler = shared_vectorizer.transform([codes_filler[i] for i in train_idx])
X_test_filler = shared_vectorizer.transform([codes_filler[i] for i in test_idx])
y_train_arr = [binary_labels[i] for i in train_idx]
y_test_arr = [binary_labels[i] for i in test_idx]

clf_filler = LogisticRegression(max_iter=1000, C=1.0)
clf_filler.fit(X_train_filler, y_train_arr)
acc_filler = accuracy_score(y_test_arr, clf_filler.predict(X_test_filler))
marker_coef_filler = float(clf_filler.coef_[0][marker_idx]) if marker_idx is not None else None
print(f"\n--- Original filler arm ---")
print(f"  Accuracy: {acc_filler:.4f}")
print(f"  Marker (' # ') coefficient: {marker_coef_filler:+.3f}")

# ---------- TEST 1: Permutation test ----------
# Break the length<->label correlation: independently permute WHICH
# document's original comment-length each document borrows for its
# filler length target, so the filler length distribution stays
# realistic but is no longer tied to that document's own label.
def length_permuted_filler(codes, binary_labels, seed=42):
    rng = np.random.RandomState(seed)
    filler_vocab = [
        'note', 'todo', 'section', 'block', 'helper', 'value', 'result',
        'item', 'entry', 'field', 'param', 'check', 'update', 'process',
        'handle', 'compute', 'return', 'temp', 'buffer', 'loop', 'case',
        'default', 'config', 'option', 'state', 'step', 'part', 'group',
    ]
    n = len(codes)
    orig_lens = []
    code_onlys = []
    for c in codes:
        code_only, comment_text = extract_comment_blocks(c)
        code_onlys.append(code_only)
        orig_lens.append(len(comment_text))
    # Permute the length assignment across ALL documents (independent of
    # label), so doc i borrows doc perm[i]'s original comment length as
    # its filler target length instead of its own.
    perm = rng.permutation(n)
    permuted_target_lens = [orig_lens[perm[i]] for i in range(n)]

    out = []
    for i in range(n):
        target_len = permuted_target_lens[i]
        filler = []
        length = 0
        while length < target_len:
            word = filler_vocab[rng.randint(len(filler_vocab))]
            filler.append(word)
            length += len(word) + 1
        out.append(code_onlys[i] + '\n# ' + ' '.join(filler))
    return out

codes_perm = length_permuted_filler(codes, binary_labels, seed=42)
X_train_perm = shared_vectorizer.transform([codes_perm[i] for i in train_idx])
X_test_perm = shared_vectorizer.transform([codes_perm[i] for i in test_idx])

clf_perm = LogisticRegression(max_iter=1000, C=1.0)
clf_perm.fit(X_train_perm, y_train_arr)
acc_perm = accuracy_score(y_test_arr, clf_perm.predict(X_test_perm))
marker_coef_perm = float(clf_perm.coef_[0][marker_idx]) if marker_idx is not None else None

print(f"\n--- TEST 1: Permutation (length<->label correlation broken) ---")
print(f"  Accuracy: {acc_perm:.4f}  (was {acc_filler:.4f})")
print(f"  Marker (' # ') coefficient: {marker_coef_perm:+.3f}  (was {marker_coef_filler:+.3f})")
collapse_pct = 100 * (1 - abs(marker_coef_perm) / abs(marker_coef_filler)) if marker_coef_filler else None
print(f"  Coefficient magnitude change: {collapse_pct:+.1f}%")
if marker_coef_perm is not None and marker_coef_filler is not None:
    if abs(marker_coef_perm) < 0.3 * abs(marker_coef_filler):
        print("  -> COLLAPSES toward zero: confirms marker's signal comes from length<->label correlation")
    else:
        print("  -> Does NOT collapse: marker retains predictive power independent of the length<->label link")

# ---------- TEST 2: Marker ablation ----------
# Drop the " # " column entirely (zero it out) and refit/re-evaluate.
if marker_idx is not None:
    import scipy.sparse as sp
    def zero_marker_column(X, idx):
        X = X.tolil()
        X[:, idx] = 0
        return X.tocsr()

    X_train_noablate = zero_marker_column(X_train_filler, marker_idx)
    X_test_noablate = zero_marker_column(X_test_filler, marker_idx)

    clf_noablate = LogisticRegression(max_iter=1000, C=1.0)
    clf_noablate.fit(X_train_noablate, y_train_arr)
    acc_noablate = accuracy_score(y_test_arr, clf_noablate.predict(X_test_noablate))

    print(f"\n--- TEST 2: Marker ablation (zero out ' # ' column, refit) ---")
    print(f"  Accuracy without marker: {acc_noablate:.4f}")
    print(f"  Accuracy with marker:    {acc_filler:.4f}")
    print(f"  Marker's contribution:   {acc_filler - acc_noablate:+.4f} ({(acc_filler-acc_noablate)*100:.2f}pp)")
else:
    acc_noablate = None

results = {
    'filler_arm_original': {'accuracy': round(acc_filler, 4), 'marker_coef': round(marker_coef_filler, 4) if marker_coef_filler else None},
    'test1_permutation': {
        'accuracy': round(acc_perm, 4),
        'marker_coef': round(marker_coef_perm, 4) if marker_coef_perm else None,
        'coef_magnitude_change_pct': round(collapse_pct, 1) if collapse_pct is not None else None,
        'collapses': bool(abs(marker_coef_perm) < 0.3 * abs(marker_coef_filler)) if marker_coef_perm and marker_coef_filler else None,
    },
    'test2_marker_ablation': {
        'accuracy_without_marker': round(acc_noablate, 4) if acc_noablate is not None else None,
        'accuracy_with_marker': round(acc_filler, 4),
        'marker_contribution_pp': round((acc_filler - acc_noablate) * 100, 2) if acc_noablate is not None else None,
    },
}
with open('howcani_falsification_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to howcani_falsification_test.json")
