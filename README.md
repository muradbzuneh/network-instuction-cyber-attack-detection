# network-instuction-cyber-attack-detection

Smart firewall training pipeline for binary intrusion detection (normal traffic vs. cyber-attack traffic such as DDoS, port scans, and brute-force attempts).

## Dataset
Use a CSV export from one of these datasets:
- CIC-IDS (Canadian Institute for Cybersecurity)
- UNSW-NB15

Expected data characteristics:
- Includes a binary or categorical label column (`label`, `attack_cat`, `class`, or `target` by default)
- Includes numeric traffic features (UNSW-NB15 style datasets include many numerical features)

## Train the model
```bash
python train_smart_firewall.py --data /absolute/path/to/dataset.csv
```

Optional tuning:
```bash
python train_smart_firewall.py \
  --data /absolute/path/to/dataset.csv \
  --label-column label \
  --pca-variance 0.95 \
  --beta 1.5 \
  --model-out smart_firewall_model.joblib
```

## What the training pipeline does
- **Feature scaling** with `StandardScaler`
- **Dimensionality reduction** with `PCA`
- **Precision vs. Recall optimization** via threshold tuning using **F-beta** on validation data (`beta > 1` prioritizes recall, `beta < 1` prioritizes precision)

The script prints JSON metrics and a test classification report, then saves a model bundle (pipeline + chosen threshold) to disk.
