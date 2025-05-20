import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary_from_sapunet_classifier(classifier, embedding_2d, labels, output_path, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = len(np.unique(labels))
    x_min, x_max = embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1
    y_min, y_max = embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    classifier.eval()
    with torch.no_grad():
        Z = torch.argmax(classifier(grid_points), axis=1).numpy().reshape(xx.shape)

    cmap_background = ListedColormap(['#F08080', '#90EE90', '#87CEFA', '#FFD700', '#DA70D6'])
    cmap_points = ListedColormap(['#B22222', '#228B22', '#1E90FF', '#FFA500', '#8A2BE2'])

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background, levels=np.arange(num_classes + 1) - 0.5)
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap_points, edgecolor='k', s=20)
    plt.colorbar(scatter, label="Class Labels")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("SAPUNet Classifier Decision Boundary in Latent Space")
    plt.savefig(output_path)
    plt.close()

def plot_unknown_on_existing_decision_space(embedding_2d, labels, unknown_embedding, output_path, seed=42):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    class DNN(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, 64)
            self.fc2 = torch.nn.Linear(64, 64)
            self.fc3 = torch.nn.Linear(64, num_classes)
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = len(np.unique(labels))
    model = DNN(input_dim=2, num_classes=num_classes)
    model.train()

    X_train, X_test, y_train, y_test = train_test_split(embedding_2d, labels, test_size=0.2, random_state=seed)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    x_min, x_max = embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1
    y_min, y_max = embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = torch.argmax(model(grid), axis=1).numpy().reshape(xx.shape)

    cmap_background = ListedColormap(['#F08080', '#90EE90', '#87CEFA', '#FFD700', '#DA70D6'])
    cmap_points = ListedColormap(['#B22222', '#228B22', '#1E90FF', '#FFA500', '#8A2BE2'])

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background, levels=np.arange(num_classes + 1) - 0.5)
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap_points, edgecolor='k', s=20)
    plt.scatter(unknown_embedding[:, 0], unknown_embedding[:, 1], c='red', marker='x', s=60, label="Unknown Samples")
    plt.colorbar(scatter, label="Class Labels")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title("Latent Space with Decision Boundary and Unknown Samples")
    plt.legend()
    plt.savefig(output_path)
    plt.close()
