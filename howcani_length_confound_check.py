#!/usr/bin/env python3
"""
Answering howcani's Sep 13 self-correction + 3 requested checks:

  1. In the permuted arm: mean marker weight by label + AUC of marker
     weight alone predicting label. If AUC stays clearly above 0.5, the
     residue is real signal, not noise.
  2. corr(len(code with comments stripped), label) on the actual split.
     Non-zero means a second channel exists (code length itself
     correlates with label) and the marker is at least partly a proxy
     for THAT, not comment content.
  3. DECISIVE: refit the filler arm with log(document length in chars)
     as an explicit feature. If the marker's coefficient collapses once
     document length is explicitly modeled, it was a norm/collinearity
     proxy all along, not a real leak and not a neutral control either.

Reuses the same shared-vectorizer pipeline as leakage_followup.py /
howcani_falsification_test.py so results are directly comparable.
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/vulndetect')
import numpy as np
import json
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from leakage_followup import extract_comment_blocks, length_matched_filler
from howcani_falsification_test import length_permuted_filler

print("Loading dataset...")
ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
codes = [x['code'] for x in ds['train']]
labels = [x['label'] for x in ds['train']]
binary_labels = [0 if l == 'safe' else 1 for l in labels]

train_idx, test_idx = train_test_split(
    list(range(len(codes))), test_size=0.2, random_state=42, stratify=binary_labels)
y_train_arr = [binary_labels[i] for i in train_idx]
y_test_arr = [binary_labels[i] for i in test_idx]

X_train_baseline_codes = [codes[i] for i in train_idx]

shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                     max_features=50000, sublinear_tf=True)
shared_vectorizer.fit(X_train_baseline_codes)
feature_names = list(shared_vectorizer.get_feature_names_out())
marker = " # "
marker_idx = feature_names.index(marker) if marker in feature_names else None
print(f"Marker index: {marker_idx}")

# ---------- Build the arms ----------
codes_filler = length_matched_filler(codes, seed=42)
codes_perm = length_permuted_filler(codes, binary_labels, seed=42)

X_train_filler = shared_vectorizer.transform([codes_filler[i] for i in train_idx])
X_test_filler = shared_vectorizer.transform([codes_filler[i] for i in test_idx])
X_train_perm = shared_vectorizer.transform([codes_perm[i] for i in train_idx])
X_test_perm = shared_vectorizer.transform([codes_perm[i] for i in test_idx])

clf_filler = LogisticRegression(max_iter=1000, C=1.0)
clf_filler.fit(X_train_filler, y_train_arr)
acc_filler = accuracy_score(y_test_arr, clf_filler.predict(X_test_filler))
marker_coef_filler = float(clf_filler.coef_[0][marker_idx])

clf_perm = LogisticRegression(max_iter=1000, C=1.0)
clf_perm.fit(X_train_perm, y_train_arr)
acc_perm = accuracy_score(y_test_arr, clf_perm.predict(X_test_perm))
marker_coef_perm = float(clf_perm.coef_[0][marker_idx])

# ---------- CHECK 1: marker-weight AUC by label, in the permuted arm ----------
marker_weight_test_perm = np.asarray(X_test_perm[:, marker_idx].todense()).ravel()
y_test_np = np.array(y_test_arr)

mean_weight_label0 = float(marker_weight_test_perm[y_test_np == 0].mean())
mean_weight_label1 = float(marker_weight_test_perm[y_test_np == 1].mean())
auc_marker_alone_perm = float(roc_auc_score(y_test_np, marker_weight_test_perm))

print(f"\n--- CHECK 1: marker weight by label, PERMUTED arm ---")
print(f"  Mean marker weight | label=0 (safe): {mean_weight_label0:.4f}")
print(f"  Mean marker weight | label=1 (vuln): {mean_weight_label1:.4f}")
print(f"  AUC(marker weight alone, label):      {auc_marker_alone_perm:.4f}")
print(f"  (howcani's threshold: AUC clearly above 0.5, e.g. ~0.65, means residue is real)")

# Also compute the same for the original (linked) filler arm, for comparison
marker_weight_test_filler = np.asarray(X_test_filler[:, marker_idx].todense()).ravel()
auc_marker_alone_filler = float(roc_auc_score(y_test_np, marker_weight_test_filler))
print(f"\n  For comparison, original (linked) filler arm:")
print(f"  AUC(marker weight alone, label):      {auc_marker_alone_filler:.4f}")

# ---------- CHECK 2: corr(len(code with comments stripped), label) ----------
code_lengths = []
for i in test_idx:
    code_only, _ = extract_comment_blocks(codes[i])
    code_lengths.append(len(code_only))
code_lengths = np.array(code_lengths, dtype=float)

corr_codelen_label = float(np.corrcoef(code_lengths, y_test_np)[0, 1])
mean_codelen_label0 = float(code_lengths[y_test_np == 0].mean())
mean_codelen_label1 = float(code_lengths[y_test_np == 1].mean())

print(f"\n--- CHECK 2: corr(code-with-comments-stripped length, label) ---")
print(f"  Mean code length | label=0 (safe): {mean_codelen_label0:.1f} chars")
print(f"  Mean code length | label=1 (vuln): {mean_codelen_label1:.1f} chars")
print(f"  Pearson r: {corr_codelen_label:+.4f}")
print(f"  (howcani: non-zero means a second channel exists -- the marker may be")
print(f"   partly a proxy for CODE length, not just comment length)")

# ---------- CHECK 3 (DECISIVE): refit filler arm with log(doc length) as explicit feature ----------
def doc_lengths_for(idx_list, code_list):
    return np.array([len(code_list[i]) for i in idx_list], dtype=float)

train_doc_lens = doc_lengths_for(train_idx, codes_filler)
test_doc_lens = doc_lengths_for(test_idx, codes_filler)
log_train_len = np.log1p(train_doc_lens).reshape(-1, 1)
log_test_len = np.log1p(test_doc_lens).reshape(-1, 1)

X_train_with_len = sp.hstack([X_train_filler, sp.csr_matrix(log_train_len)]).tocsr()
X_test_with_len = sp.hstack([X_test_filler, sp.csr_matrix(log_test_len)]).tocsr()

clf_with_len = LogisticRegression(max_iter=1000, C=1.0)
clf_with_len.fit(X_train_with_len, y_train_arr)
acc_with_len = accuracy_score(y_test_arr, clf_with_len.predict(X_test_with_len))
marker_coef_with_len = float(clf_with_len.coef_[0][marker_idx])
loglen_coef = float(clf_with_len.coef_[0][-1])

print(f"\n--- CHECK 3 (DECISIVE): refit filler arm with log(doc length) as explicit feature ---")
print(f"  Accuracy: {acc_with_len:.4f}  (was {acc_filler:.4f} without the length feature)")
print(f"  Marker (' # ') coefficient: {marker_coef_with_len:+.3f}  (was {marker_coef_filler:+.3f} without length feature)")
print(f"  log(doc length) coefficient: {loglen_coef:+.3f}")
collapse_pct_decisive = 100 * (1 - abs(marker_coef_with_len) / abs(marker_coef_filler))
print(f"  Marker coefficient change: {collapse_pct_decisive:+.1f}%")
if abs(marker_coef_with_len) < 0.3 * abs(marker_coef_filler):
    verdict = "COLLAPSES: marker was a document-norm proxy for length, not an independent leak or neutral control -- reclassify ' # ' as collinearity."
else:
    verdict = "Does NOT collapse: marker retains coefficient mass even with document length explicitly modeled -- some independent signal remains unexplained by length alone."
print(f"  -> {verdict}")

results = {
    'check1_marker_auc_permuted_arm': {
        'mean_weight_label0_safe': round(mean_weight_label0, 4),
        'mean_weight_label1_vuln': round(mean_weight_label1, 4),
        'auc_marker_alone': round(auc_marker_alone_perm, 4),
        'auc_marker_alone_original_linked_arm_for_comparison': round(auc_marker_alone_filler, 4),
    },
    'check2_code_length_vs_label_correlation': {
        'mean_code_length_label0_safe': round(mean_codelen_label0, 1),
        'mean_code_length_label1_vuln': round(mean_codelen_label1, 1),
        'pearson_r': round(corr_codelen_label, 4),
    },
    'check3_decisive_log_length_refit': {
        'accuracy_with_length_feature': round(acc_with_len, 4),
        'accuracy_without_length_feature': round(acc_filler, 4),
        'marker_coef_with_length_feature': round(marker_coef_with_len, 4),
        'marker_coef_without_length_feature': round(marker_coef_filler, 4),
        'log_length_coef': round(loglen_coef, 4),
        'marker_coef_change_pct': round(collapse_pct_decisive, 1),
        'verdict': verdict,
    },
}
with open('howcani_length_confound_check.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to howcani_length_confound_check.json")
