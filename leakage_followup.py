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


def run_experiment(codes, binary_labels, name, description, vectorizer=None):
    """Train and evaluate TF-IDF + LR on given data.

    If `vectorizer` is passed in already fitted, reuse it (transform only)
    instead of refitting — refitting per condition means each condition
    gets its own vocabulary/idf, so accuracy differences conflate feature-
    space drift with the actual content change being tested. Caught by a
    dev.to commenter (howcani) on 2026-09-11; confirmed real by inspection.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                      max_features=50000, sublinear_tf=True)
        X_train_vec = vectorizer.fit_transform(X_train)
    else:
        X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    tfidf = vectorizer
    
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
        'vectorizer': tfidf,
        'y_test': list(y_test),
        'y_pred': list(y_pred),
        'correct': [int(a == b) for a, b in zip(y_test, y_pred)],
    }


def extract_comment_blocks(code):
    """Pull out comment/docstring text and return (code_with_comments_removed, comment_text)."""
    comments = []
    def collect(pattern, text):
        found = re.findall(pattern, text)
        comments.extend(found)
        return re.sub(pattern, '', text)
    remaining = code
    remaining = collect(r'#[^\n]*', remaining)
    remaining = collect(r'//[^\n]*', remaining)
    remaining = collect(r'/\*[\s\S]*?\*/', remaining)
    remaining = collect(r'"""[\s\S]*?"""', remaining)
    remaining = collect(r"'''[\s\S]*?'''", remaining)
    return remaining, ' '.join(comments)


def cross_document_shuffle(codes, seed=42):
    """Strip comments from every doc, then reattach a RANDOM OTHER doc's
    comment block. Preserves per-document length/structure distribution
    (every doc still gets a comment block of roughly comment-shaped text)
    while destroying the comment-to-label association — unlike deleting
    comments outright, which confounds "no comment signal" with "shorter,
    differently-structured input" (howcani's point).
    """
    rng = np.random.RandomState(seed)
    stripped_codes = []
    comment_blocks = []
    for c in codes:
        code_only, comment_text = extract_comment_blocks(c)
        stripped_codes.append(code_only)
        comment_blocks.append(comment_text)
    n = len(codes)
    # Derangement-ish shuffle: permute indices so (with high probability)
    # no document gets its own original comment block back.
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    if len(fixed) > 1:
        perm[fixed] = perm[np.roll(fixed, 1)]
    shuffled = [stripped_codes[i] + '\n' + comment_blocks[perm[i]] for i in range(n)]
    return shuffled


def length_matched_filler(codes, seed=42):
    """Strip comments, then reattach filler text of roughly the same
    length as the original comment block — drawn from natural-language
    words that are neither code tokens nor label-adjacent — so length is
    matched without any real comment signal at all. Splits the effect
    three ways: stripped vs filler isolates pure length/structure effect,
    filler vs shuffled isolates real-comment-content effect.
    """
    rng = np.random.RandomState(seed)
    filler_vocab = [
        'note', 'todo', 'section', 'block', 'helper', 'value', 'result',
        'item', 'entry', 'field', 'param', 'check', 'update', 'process',
        'handle', 'compute', 'return', 'temp', 'buffer', 'loop', 'case',
        'default', 'config', 'option', 'state', 'step', 'part', 'group',
    ]
    out = []
    for c in codes:
        code_only, comment_text = extract_comment_blocks(c)
        target_len = len(comment_text)
        filler = []
        length = 0
        while length < target_len:
            word = filler_vocab[rng.randint(len(filler_vocab))]
            filler.append(word)
            length += len(word) + 1
        out.append(code_only + '\n# ' + ' '.join(filler))
    return out


def paired_delta(baseline_result, other_result, label):
    """Per-document paired accuracy delta between two conditions run on
    the SAME test indices (guaranteed here since train_test_split uses
    the same random_state/stratify on same-length/same-order code lists).
    Reports mean paired delta + 95% CI via simple bootstrap, not just a
    single accuracy-point difference (howcani's point: a 2pp effect on a
    20% split is inside the noise as a point estimate).
    """
    base_correct = np.array(baseline_result['correct'])
    other_correct = np.array(other_result['correct'])
    assert len(base_correct) == len(other_correct), "test sets must align"
    diffs = base_correct.astype(float) - other_correct.astype(float)
    rng = np.random.RandomState(0)
    n = len(diffs)
    boot_means = [diffs[rng.randint(0, n, n)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean_delta = diffs.mean()
    print(f"\n  Paired delta ({label}): baseline correct-rate minus other correct-rate")
    print(f"    mean = {mean_delta:+.4f}  (95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}])")
    if lo <= 0 <= hi:
        print(f"    -> CI crosses zero: NOT distinguishable from noise at this sample size")
    else:
        print(f"    -> CI excludes zero: effect is real at 95% confidence")
    return {'mean_delta': round(float(mean_delta), 4), 'ci_lo': round(float(lo), 4), 'ci_hi': round(float(hi), 4)}


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

    # Fit ONE vectorizer on the baseline training split, reuse it for every
    # condition below. Previously each condition refit its own vectorizer,
    # so accuracy deltas conflated feature-space drift (different vocab/idf
    # per condition) with the actual content change being tested.
    X_train_baseline, _, _, _ = train_test_split(
        codes, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
    shared_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6),
                                         max_features=50000, sublinear_tf=True)
    shared_vectorizer.fit(X_train_baseline)

    # Baseline: original (with comments)
    all_results['baseline'] = run_experiment(
        codes, binary_labels,
        "Baseline (original)",
        "Raw code with all comments and docstrings intact",
        vectorizer=shared_vectorizer
    )
    
    # Experiment 1: Strip comments
    codes_stripped = [strip_all_comments(c) for c in codes]
    all_results['no_comments'] = run_experiment(
        codes_stripped, binary_labels,
        "Comments stripped",
        "All comments, docstrings, and pure-text lines removed",
        vectorizer=shared_vectorizer
    )
    
    # Experiment 2: Strip label words
    codes_no_labels = [strip_label_words(c) for c in codes]
    all_results['no_label_words'] = run_experiment(
        codes_no_labels, binary_labels,
        "Label words removed",
        "Words like 'vulnerable', 'safe', 'injection' etc removed",
        vectorizer=shared_vectorizer
    )
    
    # Experiment 3: Both
    codes_both = [strip_label_words(strip_all_comments(c)) for c in codes]
    all_results['no_comments_no_labels'] = run_experiment(
        codes_both, binary_labels,
        "Comments stripped + label words removed",
        "Both comment removal and label word removal",
        vectorizer=shared_vectorizer
    )
    
    # Experiment 4: Normalize identifiers too
    codes_full_clean = [normalize_identifiers(strip_label_words(strip_all_comments(c))) for c in codes]
    all_results['full_clean'] = run_experiment(
        codes_full_clean, binary_labels,
        "Full clean (comments + labels + identifiers)",
        "Comments stripped, label words removed, identifiers normalized",
        vectorizer=shared_vectorizer
    )

    # Experiment 5: Cross-document comment shuffle — decomposes the
    # "no_comments" drop into a length/structure effect vs a real
    # comment-content effect, per howcani's dev.to comment (2026-09-11).
    codes_shuffled = cross_document_shuffle(codes, seed=42)
    all_results['comments_shuffled'] = run_experiment(
        codes_shuffled, binary_labels,
        "Comments shuffled across documents",
        "Own comments removed, replaced with another random doc's comment block "
        "(preserves length/structure, destroys comment-to-label association)",
        vectorizer=shared_vectorizer
    )

    # Experiment 6: Length-matched filler — same length as original comment
    # block, zero real comment signal. Splits the effect three ways:
    # stripped vs filler = pure length/structure; filler vs shuffled = real
    # comment-content effect.
    codes_filler = length_matched_filler(codes, seed=42)
    all_results['comments_filler'] = run_experiment(
        codes_filler, binary_labels,
        "Comments replaced with length-matched filler",
        "Own comments removed, replaced with neutral filler text matched to "
        "original comment length (isolates pure length/structure effect)",
        vectorizer=shared_vectorizer
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

    # Decomposition: length/structure effect vs real comment-content effect,
    # reported as paired per-document deltas with bootstrap CIs rather than
    # single accuracy-point differences (howcani: 2pp on 20% split is
    # inside the noise as a point estimate).
    print("\n" + "="*60)
    print("DECOMPOSITION: Length/Structure Effect vs Real Comment-Content Effect")
    print("="*60)
    decomposition = {}
    decomposition['baseline_vs_stripped'] = paired_delta(
        all_results['baseline'], all_results['no_comments'],
        "baseline vs stripped (combined length+content effect)")
    decomposition['baseline_vs_filler'] = paired_delta(
        all_results['baseline'], all_results['comments_filler'],
        "baseline vs length-matched filler (pure length/structure effect)")
    decomposition['filler_vs_shuffled'] = paired_delta(
        all_results['comments_filler'], all_results['comments_shuffled'],
        "filler vs shuffled real comments (real comment-content effect, length held constant)")

    # Strip non-JSON-serializable vectorizer objects and raw y arrays before dumping
    for r in all_results.values():
        r.pop('vectorizer', None)
        r.pop('y_test', None)
        r.pop('y_pred', None)
        r.pop('correct', None)
    all_results['decomposition'] = decomposition

    with open('leakage_followup_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to leakage_followup_results.json")


if __name__ == '__main__':
    main()
