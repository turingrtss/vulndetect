#!/usr/bin/env python3
"""
Answering howcani's Sep 14 reply: paired McNemar test on discordant cells
for baseline (with real comments) vs stripped (code-only, no comments).
Same held-out test set, same shared vectorizer -- this is the paired
comparison they asked for, done for BOTH the real-comment arm and the
permuted (filler) arm, so both can be checked for separability at this n.

Report: both-correct / comment-only-correct / stripped-only-correct /
both-wrong counts, plus exact McNemar p (binomial exact on discordant
pairs, since n is small enough that the chi-square approx is shaky).
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/vulndetect')
import numpy as np
import json
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest
from leakage_followup import extract_comment_blocks
from howcani_falsification_test import length_permuted_filler

print("Loading dataset...")
ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
codes = [x['code'] for x in ds['train']]
labels = [x['label'] for x in ds['train']]
binary_labels = [0 if l == 'safe' else 1 for l in labels]

train_idx, test_idx = train_test_split(
    list(range(len(codes))), test_size=0.2, random_state=42, stratify=binary_labels)
y_train_arr = np.array([binary_labels[i] for i in train_idx])
y_test_arr = np.array([binary_labels[i] for i in test_idx])

X_train_baseline_codes = [codes[i] for i in train_idx]
shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                     max_features=50000, sublinear_tf=True)
shared_vectorizer.fit(X_train_baseline_codes)

def mask_comment_region(code):
    code_only, _ = extract_comment_blocks(code)
    return code_only

codes_baseline = codes
codes_code_only = [mask_comment_region(c) for c in codes]
codes_perm = length_permuted_filler(codes, binary_labels, seed=42)

def fit_predict(code_list):
    Xtr = shared_vectorizer.transform([code_list[i] for i in train_idx])
    Xte = shared_vectorizer.transform([code_list[i] for i in test_idx])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, y_train_arr)
    return clf.predict(Xte)

pred_baseline = fit_predict(codes_baseline)
pred_stripped = fit_predict(codes_code_only)
pred_perm = fit_predict(codes_perm)

def mcnemar_report(pred_a, pred_b, y, label_a, label_b):
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)
    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    b_only = int(np.sum(~correct_a & correct_b))   # b right, a wrong
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    n_discordant = a_only + b_only
    if n_discordant > 0:
        # exact two-sided binomial test on discordant pairs, p=0.5 under null
        res = binomtest(min(a_only, b_only), n_discordant, 0.5, alternative='two-sided')
        p_exact = res.pvalue
    else:
        p_exact = 1.0
    print(f"\n--- {label_a} vs {label_b} (paired, n={len(y)}) ---")
    print(f"  both correct:            {both_correct}")
    print(f"  {label_a}-only correct:  {a_only}")
    print(f"  {label_b}-only correct:  {b_only}")
    print(f"  both wrong:               {both_wrong}")
    print(f"  discordant pairs: {n_discordant}")
    print(f"  McNemar exact p (two-sided): {p_exact:.4f}")
    return {
        'both_correct': both_correct,
        f'{label_a}_only_correct': a_only,
        f'{label_b}_only_correct': b_only,
        'both_wrong': both_wrong,
        'n_discordant': n_discordant,
        'mcnemar_exact_p': round(float(p_exact), 4),
    }

real_arm = mcnemar_report(pred_baseline, pred_stripped, y_test_arr, 'baseline', 'stripped')
perm_arm = mcnemar_report(pred_perm, pred_stripped, y_test_arr, 'permuted', 'stripped')

results = {
    'real_comment_arm_vs_stripped': real_arm,
    'permuted_filler_arm_vs_stripped': perm_arm,
    'note': 'Same held-out test set, same shared vectorizer (fit on baseline train split) for all three conditions -- paired comparison, McNemar exact binomial on discordant cells.',
}
with open('howcani_mcnemar_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to howcani_mcnemar_test.json")
