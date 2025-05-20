import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import json
import random
from utils.plot_utils import plot_decision_boundary_from_sapunet_classifier, plot_unknown_on_existing_decision_space
from utils.eval_metrics import compute_clustering_metrics


# ---------- Model Definition ----------
class ParametricUMAPEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class MLPClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, z):
        return self.classifier(z)
    

# ---------- Ealy Stopping ----------
class EarlyStopping:
    def __init__(self, patience=20, delta=0.001, verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, encoder, classifier):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(encoder, classifier)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(encoder, classifier)
            self.counter = 0

    def save_checkpoint(self, encoder, classifier):
        torch.save(encoder.state_dict(), self.path.replace('.pt', '_encoder.pt'))
        torch.save(classifier.state_dict(), self.path.replace('.pt', '_classifier.pt'))



# ---------- UMAP Graph Loss ----------
def compute_umap_graph_loss(z, original_data, n_neighbors=15, a=1.929, b=0.791):
    N = z.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(original_data)
    distances, indices = nbrs.kneighbors(original_data)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    edge_i = np.repeat(np.arange(N), n_neighbors)
    edge_j = indices.flatten()
    edge_d = distances.flatten()
    p_ij = np.exp(-edge_d)
    p_ij = torch.tensor(p_ij, dtype=torch.float32, device=z.device)
    zi = z[edge_i]
    zj = z[edge_j]
    embedding_dist = torch.norm(zi - zj, dim=1)
    q_ij = (1 + a * embedding_dist.pow(2)) ** (-b)
    eps = 1e-4
    loss = -torch.mean(p_ij * torch.log(q_ij + eps) + (1 - p_ij) * torch.log(1 - q_ij + eps))
    return loss



# ---------- Classification Metrics ----------
def classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        return o

    return {"accuracy": float(acc), "report": convert(report), "confusion_matrix": convert(cm)}



# ---------- ConfigurableGeoUMAPNet ----------
class ConfigurableGeoUMAPNet:
    def __init__(self, input_dim, latent_dim, num_classes, use_umap_loss=True):
        self.encoder = ParametricUMAPEncoder(input_dim, latent_dim)
        self.classifier = MLPClassifier(latent_dim, num_classes)
        self.use_umap_loss = use_umap_loss
        self.ce_loss = nn.CrossEntropyLoss()

    def train_step(self, X_tensor, y_tensor, original_data, optimizer, coords=None, lambda_smooth=0.3):
        self.encoder.train(); self.classifier.train()
        z = self.encoder(X_tensor)
        preds = self.classifier(z)
        ce = self.ce_loss(preds, y_tensor)

        if self.use_umap_loss:
            umap = compute_umap_graph_loss(z, original_data)
        else:
            umap = torch.tensor(0.0)

        if coords is not None:
            smooth = debug_spatial_smoothness(z, coords, k=8, sigma=2000)
            assert z.shape[0] == coords.shape[0], "Mismatch between z and coords lengths"
        else:
            smooth = torch.tensor(0.0)

        loss = ce + umap + lambda_smooth * smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), ce.item(), umap.item(), smooth.item()




# ---------- Unkown Sample Prediction ----------
def predict_unknown(df, encoder, classifier, scaler, output_csv="outputs/predicted_unknowns.csv", feature_columns=None):
    unknown_df = df[df["label"].isna()].copy()
    unknown_features = df.loc[unknown_df.index, feature_columns]
    X = scaler.transform(unknown_features)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    encoder.eval(); classifier.eval()
    with torch.no_grad():
        z = encoder(X_tensor)
        probs = torch.softmax(classifier(z), dim=1).numpy()
        pred_labels = np.argmax(probs, axis=1)

    unknown_df["predicted_label"] = pred_labels
    for i in range(probs.shape[1]):
        unknown_df[f"prob_class_{i}"] = probs[:, i]
    unknown_df.to_csv(output_csv, index=False)
    print(f"Unkonwn samples predicted and saved to：{output_csv}")



# ---------- main ----------
def run_experiment(csv_path, output_dir="outputs", latent_dim=2, num_classes=4,
                   epochs=1000, lr=1e-3, test_size=0.3, use_umap_loss=True,
                   patience=20):
    set_random_seed(42)
    df = pd.read_csv(csv_path)
    features = df.drop(columns=['y11', 'x11', 'Y1', 'X1', 'y', 'x', 'FID', 'ID', 'label','Location'], errors='ignore')
    labels = df['label']
    labeled_data = features[labels.notna()]
    labeled_labels = labels[labels.notna()].astype(int)
    feature_columns = labeled_data.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(labeled_data)

    coords_all = df.loc[labeled_data.index, ['xx', 'yy']].to_numpy()

    X_train_full, X_test, y_train_full, y_test, coords_train_full, coords_test = train_test_split(
        X_scaled, labeled_labels, coords_all, test_size=test_size,
        stratify=labeled_labels, random_state=42
    )

    X_train, X_val, y_train, y_val, coords_train, coords_val = train_test_split(
        X_train_full, y_train_full, coords_train_full, test_size=0.2,
        stratify=y_train_full, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    model = ConfigurableGeoUMAPNet(X_train.shape[1], latent_dim, num_classes, use_umap_loss)
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=lr
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=os.path.join(output_dir, 'best_model.pt'))
    loss_log = []

    for epoch in range(epochs):
        loss, ce, umap, smooth = model.train_step(
            X_train_tensor, y_train_tensor, X_train,
            optimizer, coords=coords_train, lambda_smooth=0.5
        )

        model.encoder.eval(); model.classifier.eval()
        with torch.no_grad():
            z_val = model.encoder(X_val_tensor)
            val_preds = model.classifier(z_val)
            val_loss = model.ce_loss(val_preds, y_val_tensor).item()

        loss_log.append((loss, ce, umap, smooth, val_loss))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Total Loss: {loss:.4f}, CE: {ce:.4f}, "
                  f"UMAP: {umap:.4f}, Smooth: {smooth:.4f}, Val CE: {val_loss:.4f}")

        early_stopping(val_loss, model.encoder, model.classifier)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        model.encoder.load_state_dict(torch.load(os.path.join(output_dir, 'best_model_encoder.pt'),weights_only=True), strict=False)
        model.classifier.load_state_dict(torch.load(os.path.join(output_dir, 'best_model_classifier.pt'),weights_only=True), strict=False)


    plot_loss_curves(loss_log, output_dir)

    encoder, classifier = model.encoder, model.classifier
    encoder.eval(); classifier.eval()

    with torch.no_grad():
        z_all = encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        y_all = labeled_labels.values
        z_test = encoder(X_test_tensor)
        y_pred = torch.argmax(classifier(z_test), dim=1).numpy()
        metrics = classification_metrics(y_test_tensor.numpy(), y_pred)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='Spectral', s=10)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.title("Latent Space Visualization")
    plt.savefig(os.path.join(output_dir, "embedding_visualization.png"))
    plt.close()

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    torch.save(encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))
    torch.save(classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))


    predict_unknown(
        df, encoder, classifier, scaler,
        output_csv=os.path.join(output_dir, "predicted_unknowns.csv"),
        feature_columns=feature_columns
    )

    plot_decision_boundary_from_sapunet_classifier(
        classifier=classifier,        
        embedding_2d=z_all,           
        labels=y_all,                 
        output_path=os.path.join(output_dir, "decision_boundary_sapunet_classifier.png")
)


    def plot_decision_boundary_highlighted_location_from_classifier(
        embedding_2d,
        labels,
        df,
        encoder,
        classifier,  
        scaler,
        feature_columns,
        location_keywords,
        output_path="decision_boundary_sapunet_highlighted_location.png",
        seed=42
    ):
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        np.random.seed(seed)
        torch.manual_seed(seed)

        num_classes = len(np.unique(labels))

        x_min, x_max = embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1
        y_min, y_max = embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                            np.linspace(y_min, y_max, 300))
        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        classifier.eval()
        with torch.no_grad():
            preds = torch.argmax(classifier(grid_points), axis=1).numpy()
        preds = preds.reshape(xx.shape)

        cmap_background = ListedColormap(['#F08080', '#90EE90', '#87CEFA', '#FFD700', '#DA70D6'])
        cmap_points = ListedColormap(['#B22222', '#228B22', '#1E90FF', '#FFA500', '#8A2BE2'])

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, preds, alpha=0.3, cmap=cmap_background, levels=np.arange(num_classes + 1) - 0.5)
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap_points, edgecolor='k', s=20)

        
        highlight_colors = ['black', 'lime', 'cyan', 'magenta', 'orange']
        for idx, keyword in enumerate(location_keywords):
            subset = df[df['Location'].str.contains(keyword, case=False, na=False)]
            if subset.empty:
                print(f"not found '{keyword}' samples in the dataset")
                continue
            X_sub = scaler.transform(subset[feature_columns])
            X_tensor = torch.tensor(X_sub, dtype=torch.float32)
            with torch.no_grad():
                z_sub = encoder(X_tensor).numpy()
            plt.scatter(z_sub[:, 0], z_sub[:, 1], color=highlight_colors[idx % len(highlight_colors)],
                        s=300, edgecolor='white', marker='*', label=f"{keyword}")

        plt.legend()
        plt.colorbar(scatter, label="Class Labels")
        plt.title("GeoUMAP Latent Space with SAPUNet Classifier and Highlighted Locations")
        plt.xlabel("Latent Dim 1")
        plt.ylabel("Latent Dim 2")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ SAPUNet decision boundary with highlighted locations saved to：{output_path}")


    plot_decision_boundary_highlighted_location_from_classifier(
        embedding_2d=z_all,
        labels=y_all,
        df=df,
        encoder=encoder,
        classifier=classifier, 
        scaler=scaler,
        feature_columns=feature_columns,
        location_keywords=["tiegelongnan", "naruo"],
        output_path=os.path.join(output_dir, "decision_boundary_sapunet_highlighted_location.png")
    )

    

    clustering_metrics = compute_clustering_metrics(z_all, y_all)
    print("🔍 Clustering Metrics:")
    for k, v in clustering_metrics.items():
        print(f"{k}: {v:.4f}")

     
    with torch.no_grad():
        unknown_df = df[df["label"].isna()]
        X_unknown = scaler.transform(unknown_df[feature_columns])
        z_unknown = encoder(torch.tensor(X_unknown, dtype=torch.float32)).numpy()

    plot_unknown_on_existing_decision_space(
        embedding_2d=z_all,
        labels=y_all,
        unknown_embedding=z_unknown,
        output_path=os.path.join(output_dir, "decision_boundary_with_unknowns.png")
    )



    return encoder, classifier, scaler

def debug_spatial_smoothness(z, coords, k=8, sigma=1):
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().detach().numpy()

    N = z.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    edge_i = np.repeat(np.arange(N), k)
    edge_j = indices.flatten()
    edge_w = np.exp(-distances.flatten()**2 / sigma**2)

    zi = z[edge_i]
    zj = z[edge_j]
    dist = torch.norm(zi - zj, dim=1)
    weights = torch.tensor(edge_w, dtype=torch.float32, device=z.device)
    smooth_loss = torch.mean(weights * dist**2)

    return smooth_loss

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_loss_curves(loss_log, output_dir):
    loss_log = np.array(loss_log)
    labels = ["Total Loss", "CE", "UMAP", "Smooth", "Val CE"]
    plt.figure(figsize=(10, 6))
    for i in range(loss_log.shape[1]):
        plt.plot(loss_log[:, i], label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    print(f"Loss figure saved to：{os.path.join(output_dir, 'loss_curve.png')}")


# ------------------------
encoder, classifier, scaler = run_experiment(
    csv_path="/Users/1x1ao/Library/CloudStorage/OneDrive-个人/paper_code/github集合/SAPUNet/data/your_data.csv",
    output_dir="outputs_geo",
    latent_dim=2,
    num_classes=4,
    epochs=3000,
    use_umap_loss=True,
    patience=300,
    lr=0.001
)