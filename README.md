# VulnDetect

ML-based vulnerability detection research.

## Key Finding: Label Leakage

Trained a TF-IDF classifier on a public code vulnerability dataset and got 84.7% accuracy. Feature analysis revealed the model was largely reading comments ("vulnerable", "safe") rather than learning code patterns.

**Full paper:** [paper.pdf](paper.pdf)

**Dev.to writeup:** [I Trained a Vulnerability Detection Model. It Was Reading Comments, Not Code.](https://dev.to/turingrtss/i-trained-a-vulnerability-detection-model-it-was-reading-comments-not-code-3eln)

## Results

| Model | Accuracy | F1 | FP Rate | Train Time |
|-------|----------|-----|---------|------------|
| Logistic Regression | 84.7% | 0.842 | 16.7% | 0.5s |
| Gradient Boosting | 83.7% | 0.835 | 19.5% | 175s |
| Random Forest | 82.7% | 0.821 | 17.8% | 18s |

## Reproduce

```bash
pip install numpy scikit-learn matplotlib datasets
python3 train.py
```

Runs on any machine with 4GB+ RAM. No GPU needed. Training completes in under 4 minutes.

## Next

- Strip comments/docstrings and re-evaluate
- Use CVE fix commits as clean training data
- Compare against regex-based SAST tools

## License

MIT
