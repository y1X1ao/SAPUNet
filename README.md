# SAPUNet: A Spatially-Aware Parametric UMAP Network for Interpretable Mineral Prediction and Geological Insight in the Duolong Mineral District

**Authors:**  
Yixiao Wu<sup>1</sup>, Wenlei Wang<sup>2</sup>, Changjiang Yuan<sup>1</sup>  
📧 Contact: [wuyixiao129@outlook.com](mailto:wuyixiao129@outlook.com)  

---

## 🧠 Overview

**SAPUNet** is a spatially-aware deep learning framework designed for interpretable mineral prospectivity prediction. It integrates a **parametric UMAP encoder** with a supervised classifier and incorporates **spatial smoothness** and **geological structure preservation** mechanisms. This repository presents the full implementation, as applied to geochemical data in the **Duolong mineral district**, enabling insight into mineralization patterns and model-based geological interpretation.

---

##  Key Features

-  **Parametric UMAP embedding** of high-dimensional geochemical features into 2D latent space.
-  **End-to-end training** with classification loss + UMAP loss + spatial smoothness regularization.
-  **Visual interpretation** via decision boundary plotting and highlighted geological targets .
-  **Unknown sample prediction** supported.
-  Integrated clustering metric computation (Silhouette, DBI, CH).

---

##  Project Structure

```
SAPUNet/
├── SAPUNet_main.py           # Main training, evaluation, and visualization pipeline
├── utils/
│   ├── plot_utils.py         # Plotting: decision boundaries, latent space, highlights
│   └── eval_metrics.py       # Clustering evaluation (Silhouette, DBI, CH)
├── data/                     # Input data folder (CSV format)
├── outputs_geo/              # Output folder for visualizations and predictions
├── requirements.txt          # Python dependency list
└── README.md                 # Project documentation (this file)
```

---

## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Prepare your data**:

Place your CSV dataset (e.g., `最新0517标记3.csv`) in the `data/` folder. Make sure it includes:
- `label` column: known classes (0–3) or NaN for unknowns
- `xx`, `yy` columns: spatial coordinates
- `Location` column (optional): for named highlighting

3. **Execute main pipeline**:

```bash
python sapunet_main.py
```

This will:
- Train encoder and classifier with UMAP + spatial constraints
- Save model weights
- Plot 2D latent space and decision boundaries
- Highlight known deposits 
- Predict labels for unlabeled samples

---

## 🖼️ Output Files (in `outputs_geo/`)

| File                                | Description                                      |
|-------------------------------------|--------------------------------------------------|
| `embedding_visualization.png`       | 2D latent representation of all labeled samples |
| `decision_boundary_sapunet_classifier.png` | SAPUNet classifier's decision boundary       |
| `decision_boundary_dnn_highlighted_location.png` | Same with highlighted locations         |
| `decision_boundary_with_unknowns.png` | Overlay of unknown sample projections        |
| `predicted_unknowns.csv`            | Model predictions for unknown samples           |
| `classification_report.json`        | Accuracy, confusion matrix, precision/recall    |
| `loss_curve.png`                    | Training loss components across epochs          |

---


## 📬 Contact

For questions, collaborations, or feedback, please reach out to:

**Yixiao Wu** – [wuyixiao129@outlook.com](mailto:wuyixiao129@outlook.com)

---

*Last updated: 2025*