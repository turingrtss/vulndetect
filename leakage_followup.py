#!/usr/bin/env python3
"""
Label Leakage Follow-up: Strip comments, retrain, measure the drop.

The original experiment found that a TF-IDF classifier achieved 84.7% accuracy
on a vulnerability dataset, but top features were substrings of "vulnerable" and
"safe" from code comments. This experiment strips all comments and docstrings,
retrains, and measures the true model capability after removing label leakage.

Also tests identifier normalization (replacing variable names with generic tokens).
"""

import re
import time
import json
import ast
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def strip_python_comments(code):
    """Remove comments and docstrings from Python code."""
    # Remove single-line comments
    code = re.sub(r'#[^\n]*', '', code)
    # Remove triple-quoted docstrings
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    return code


def strip_all_comments(code):
    """Remove comments from code in any language."""
    # Python comments
    code = re.sub(r'#[^\n]*', '', code)
    # C-style single line
    code = re.sub(r'//[^\n]*', '', code)
    # C-style multi-line
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Triple-quoted strings (Python docstrings)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    # Remove markdown code fences
    code = re.sub(r'```\w*\n?', '', code)
    # Remove lines that are purely natural language (no code chars)
    lines = code.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines and lines with code-like characters
        if not stripped or re.search(r'[{}\[\]();=<>+\-*/&|^~!@%:,.]', stripped):
            filtered.append(line)
        elif re.match(r'^\s*(def|class|import|from|return|if|else|elif|for|while|try|except|with|raise|yield|async|await)\b', stripped):
            filtered.append(line)
        # Skip lines that look like pure English text
    code = '\n'.join(filtered)
    return code


def normalize_identifiers(code):
    """Replace variable/function names with generic tokens."""
    # Replace common identifier patterns with generic names
    # This is a rough approximation - true normalization would need parsing
    
    # Replace function definitions
    counter = [0]
    def replace_func(match):
        counter[0] += 1
        return f"def func_{counter[0]}("
    code = re.sub(r'def\s+(\w+)\s*\(', replace_func, code)
    
    # Replace class definitions
    counter[0] = 0
    def replace_class(match):
        counter[0] += 1
        return f"class Class_{counter[0]}"
    code = re.sub(r'class\s+(\w+)', replace_class, code)
    
    # Replace string literals with generic placeholder
    code = re.sub(r'"[^"]*"', '"STR"', code)
    code = re.sub(r"'[^']*'", "'STR'", code)
    
    return code


def strip_label_words(code):
    """Remove words that directly match label categories."""
    label_words = [
        'vulnerable', 'vulnerability', 'vulnerabilities',
        'safe', 'safely', 'safety', 'secure', 'securely', 'security',
        'injection', 'overflow', 'buffer', 'exploit', 'attack',
        'malicious', 'sanitize', 'sanitization', 'insecure',
        'dangerous', 'unsafe', 'threat', 'risk', 'patch', 'fix',
    ]
    for word in label_words:
        code = re.sub(r'\b' + word + r'\b', '', code, flags=re.IGNORECASE)
    return code


def run_experiment(codes, binary_labels, name, description):
    """Train and evaluate TF-IDF + LR on given data."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
    
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                            max_features=50000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)
    train_time = time.time() - t0
    
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Top features
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_vuln = np.argsort(coefs)[-10:][::-1]
    top_safe = np.argsort(coefs)[:10]
    
    print(f"\n  Accuracy: {acc:.4f}")
    print(f"  Vuln F1:  {report['1']['f1-score']:.4f}")
    print(f"  FP Rate:  {report['0']['support'] - report['0']['recall'] * report['0']['support']:.0f}/{report['0']['support']} = {(1-report['0']['recall'])*100:.1f}%")
    print(f"  Train:    {train_time:.2f}s")
    
    print(f"\n  Top 5 VULN features:")
    for idx in top_vuln[:5]:
        print(f"    {coefs[idx]:+.3f}  '{feature_names[idx]}'")
    
    print(f"  Top 5 SAFE features:")
    for idx in top_safe[:5]:
        print(f"    {coefs[idx]:+.3f}  '{feature_names[idx]}'")
    
    return {
        'name': name,
        'accuracy': round(acc, 4),
        'vuln_f1': round(report['1']['f1-score'], 4),
        'safe_f1': round(report['0']['f1-score'], 4),
        'fp_rate': round((1 - report['0']['recall']) * 100, 1),
        'train_time': round(train_time, 2),
        'top_vuln_features': [(feature_names[i], round(float(coefs[i]), 3)) for i in top_vuln[:5]],
        'top_safe_features': [(feature_names[i], round(float(coefs[i]), 3)) for i in top_safe[:5]],
    }


def main():
    print("Label Leakage Follow-up Experiment")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
    codes = [x['code'] for x in ds['train']]
    labels = [x['label'] for x in ds['train']]
    binary_labels = [0 if l == 'safe' else 1 for l in labels]
    
    print(f"Samples: {len(codes)}")
    print(f"Safe: {binary_labels.count(0)}, Vuln: {binary_labels.count(1)}")
    
    all_results = {}
    
    # Baseline: original (with comments)
    all_results['baseline'] = run_experiment(
        codes, binary_labels,
        "Baseline (original)",
        "Raw code with all comments and docstrings intact"
    )
    
    # Experiment 1: Strip comments
    codes_stripped = [strip_all_comments(c) for c in codes]
    all_results['no_comments'] = run_experiment(
        codes_stripped, binary_labels,
        "Comments stripped",
        "All comments, docstrings, and pure-text lines removed"
    )
    
    # Experiment 2: Strip label words
    codes_no_labels = [strip_label_words(c) for c in codes]
    all_results['no_label_words'] = run_experiment(
        codes_no_labels, binary_labels,
        "Label words removed",
        "Words like 'vulnerable', 'safe', 'injection' etc removed"
    )
    
    # Experiment 3: Both
    codes_both = [strip_label_words(strip_all_comments(c)) for c in codes]
    all_results['no_comments_no_labels'] = run_experiment(
        codes_both, binary_labels,
        "Comments stripped + label words removed",
        "Both comment removal and label word removal"
    )
    
    # Experiment 4: Normalize identifiers too
    codes_full_clean = [normalize_identifiers(strip_label_words(strip_all_comments(c))) for c in codes]
    all_results['full_clean'] = run_experiment(
        codes_full_clean, binary_labels,
        "Full clean (comments + labels + identifiers)",
        "Comments stripped, label words removed, identifiers normalized"
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: How Much Was Label Leakage?")
    print("="*60)
    
    baseline_acc = all_results['baseline']['accuracy']
    
    print(f"\n  {'Condition':<45} {'Accuracy':>8} {'Drop':>8} {'Vuln F1':>8}")
    print(f"  {'-'*69}")
    for name, r in all_results.items():
        drop = baseline_acc - r['accuracy']
        print(f"  {r['name']:<45} {r['accuracy']:>8.4f} {drop:>+8.4f} {r['vuln_f1']:>8.4f}")
    
    full_drop = baseline_acc - all_results['full_clean']['accuracy']
    print(f"\n  Total accuracy attributable to label leakage: {full_drop*100:.1f} percentage points")
    print(f"  True model capability (code patterns only): {all_results['full_clean']['accuracy']*100:.1f}%")
    
    with open('leakage_followup_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to leakage_followup_results.json")


if __name__ == '__main__':
    main()
