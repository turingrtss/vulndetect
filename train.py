#!/usr/bin/env python3
"""
VulnDetect - Train a vulnerability detection model
Experiment 1: Can ML beat regex for finding code vulnerabilities?

Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU
Dataset: lemon42-ai/Code_Vulnerability_Labeled_Dataset (8,480 samples)
"""

import json
import time
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load and prepare the dataset."""
    print("Loading dataset...")
    ds = load_dataset("lemon42-ai/Code_Vulnerability_Labeled_Dataset")
    
    codes = [x['code'] for x in ds['train']]
    labels = [x['label'] for x in ds['train']]
    
    # Binary classification: vulnerable vs safe
    binary_labels = [0 if l == 'safe' else 1 for l in labels]
    
    print(f"Total samples: {len(codes)}")
    print(f"Safe: {binary_labels.count(0)}, Vulnerable: {binary_labels.count(1)}")
    print(f"Balance: {binary_labels.count(1)/len(binary_labels)*100:.1f}% vulnerable")
    
    return codes, binary_labels, labels


def train_baseline(codes, binary_labels):
    """Train TF-IDF + classifier baseline."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: TF-IDF + Classifiers (Baseline)")
    print("="*60)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels
    )
    
    # TF-IDF on code tokens
    # Use character n-grams to capture code patterns better than words
    print("\nVectorizing with TF-IDF (char n-grams 3-6)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 6),
        max_features=50000,
        sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(f"  Shape: {X_train_vec.shape}, took {time.time()-t0:.1f}s")
    
    # Try multiple classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = time.time() - t0
        
        # Predict
        y_pred = clf.predict(X_test_vec)
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report['accuracy']
        precision_vuln = report['1']['precision']
        recall_vuln = report['1']['recall']
        f1_vuln = report['1']['f1-score']
        
        # False positive rate (safe code flagged as vulnerable)
        cm = confusion_matrix(y_test, y_pred)
        fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision_vuln,
            'recall': recall_vuln,
            'f1': f1_vuln,
            'fp_rate': fp_rate,
            'train_time': train_time
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Vuln Precision: {precision_vuln:.3f} (of flagged, how many are real)")
        print(f"  Vuln Recall: {recall_vuln:.3f} (of real vulns, how many caught)")
        print(f"  Vuln F1: {f1_vuln:.3f}")
        print(f"  False Positive Rate: {fp_rate:.1f}%")
        print(f"  Train time: {train_time:.1f}s")
        print(f"\n  Full report:")
        print(classification_report(y_test, y_pred, target_names=['safe', 'vulnerable']))
    
    return results, tfidf


def train_multiclass(codes, labels):
    """Multi-class: predict specific vulnerability type."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Multi-class Vulnerability Type Detection")
    print("="*60)
    
    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    encoded = [label_to_id[l] for l in labels]
    
    X_train, X_test, y_train, y_test = train_test_split(
        codes, encoded, test_size=0.2, random_state=42, stratify=encoded
    )
    
    print(f"Classes: {len(unique_labels)}")
    
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=50000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    print("Training Logistic Regression (multi-class)...")
    t0 = time.time()
    clf.fit(X_train_vec, y_train)
    print(f"  Train time: {time.time()-t0:.1f}s")
    
    y_pred = clf.predict(X_test_vec)
    
    # Short label names for readability
    short_labels = []
    for l in unique_labels:
        if 'SQL' in l: short_labels.append('SQLi')
        elif 'Cross-site' in l: short_labels.append('XSS')
        elif 'OS Command' in l: short_labels.append('CmdInj')
        elif 'Code Injection' in l: short_labels.append('CodeInj')
        elif 'Deserialization' in l: short_labels.append('Deser')
        elif 'XML' in l: short_labels.append('XXE')
        elif 'Path Traversal' in l: short_labels.append('PathTrav')
        elif 'Open Redirect' in l: short_labels.append('OpenRedir')
        elif 'Input Validation' in l: short_labels.append('InputVal')
        elif 'Integer Overflow' in l: short_labels.append('IntOverflow')
        elif 'NULL' in l: short_labels.append('NullPtr')
        elif 'Out-of-bounds' in l: short_labels.append('OOBWrite')
        elif 'Log' in l: short_labels.append('LogInj')
        elif l == 'safe': short_labels.append('Safe')
        else: short_labels.append(l[:15])
    
    print(f"\n  Multi-class report:")
    print(classification_report(y_test, y_pred, target_names=short_labels))
    
    report = classification_report(y_test, y_pred, output_dict=True)
    return report['accuracy']


def analyze_features(tfidf, codes, binary_labels):
    """What patterns does the model think matter?"""
    print("\n" + "="*60)
    print("ANALYSIS: Top Features for Vulnerability Detection")
    print("="*60)
    
    X_vec = tfidf.transform(codes)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_vec, binary_labels)
    
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    
    # Top features indicating vulnerability
    top_vuln_idx = np.argsort(coefs)[-20:][::-1]
    print("\nTop 20 features indicating VULNERABLE code:")
    for idx in top_vuln_idx:
        print(f"  {coefs[idx]:+.3f}  '{feature_names[idx]}'")
    
    # Top features indicating safe code
    top_safe_idx = np.argsort(coefs)[:20]
    print("\nTop 20 features indicating SAFE code:")
    for idx in top_safe_idx:
        print(f"  {coefs[idx]:+.3f}  '{feature_names[idx]}'")


def main():
    print("VulnDetect - Vulnerability Detection ML Experiment")
    print("="*60)
    print(f"Hardware: ARM64, 12GB RAM, 2 CPU cores, no GPU")
    print()
    
    codes, binary_labels, labels = load_data()
    
    # Experiment 1: Binary classification
    results, tfidf = train_baseline(codes, binary_labels)
    
    # Experiment 2: Multi-class
    multiclass_acc = train_multiclass(codes, labels)
    
    # Feature analysis
    analyze_features(tfidf, codes, binary_labels)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nBinary Classification (vulnerable vs safe):")
    for name, r in results.items():
        print(f"  {name:25s}  Acc={r['accuracy']:.3f}  F1={r['f1']:.3f}  FP={r['fp_rate']:.1f}%  Time={r['train_time']:.1f}s")
    
    print(f"\nMulti-class Accuracy: {multiclass_acc:.3f}")
    
    best = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest model: {best[0]} (F1={best[1]['f1']:.3f})")
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump({
            'binary_results': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
            'multiclass_accuracy': round(multiclass_acc, 4),
            'dataset_size': len(codes),
            'hardware': 'ARM64, 12GB RAM, 2 CPU, no GPU'
        }, f, indent=2)
    
    print("\nResults saved to experiment_results.json")


if __name__ == '__main__':
    main()
