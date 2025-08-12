import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from scipy.stats import norm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Global Setup
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Data Generation Functions
def generate_easy_data(num_tasks=2000):
    """Generates dataset with clearly separated features."""
    np.random.seed(42)
    criticalities = np.random.choice(['hard', 'soft'], num_tasks, p=[0.3, 0.7])
    burst_times = np.where(
        criticalities == 'hard',
        np.random.exponential(scale=5, size=num_tasks).clip(1, 10) * np.random.uniform(0.9, 1.1, num_tasks),
        np.random.exponential(scale=100, size=num_tasks).clip(50, 200) * np.random.uniform(0.9, 1.1, num_tasks)
    )
    deadlines = np.where(
        criticalities == 'hard',
        burst_times + np.random.randint(5, 30, num_tasks) * np.random.uniform(0.9, 1.1, num_tasks),
        burst_times + np.random.randint(50, 150, num_tasks) * np.random.uniform(0.9, 1.1, num_tasks)
    )
    tasks = pd.DataFrame({
        'task_id': range(num_tasks),
        'burst_time': burst_times,
        'criticality': criticalities,
        'deadline': deadlines
    })
    tasks['criticality_num'] = tasks['criticality'].map({'hard': 1, 'soft': 0})
    tasks['urgency'] = tasks['deadline'] / tasks['burst_time'].clip(1)
    tasks['urgency'] = tasks['urgency'].clip(0, 100)
    return tasks

def generate_realistic_data(num_tasks=2000):
    """Generates realistic dataset aligned with main.py."""
    np.random.seed(None)  # Allow variation
    criticalities = np.random.choice(['hard', 'soft'], num_tasks, p=[0.3, 0.7])
    burst_times = np.where(
        criticalities == 'hard',
        np.random.exponential(scale=5, size=num_tasks).clip(1, 10) * np.random.uniform(0.9, 1.1, num_tasks),
        np.random.exponential(scale=100, size=num_tasks).clip(50, 200) * np.random.uniform(0.9, 1.1, num_tasks)
    )
    deadlines = np.where(
        criticalities == 'hard',
        burst_times + np.random.randint(5, 30, num_tasks) * np.random.uniform(0.9, 1.1, num_tasks),
        burst_times + np.random.randint(50, 150, num_tasks) * np.random.uniform(0.9, 1.1, num_tasks)
    )
    tasks = pd.DataFrame({
        'task_id': range(num_tasks),
        'burst_time': burst_times,
        'criticality': criticalities,
        'deadline': deadlines
    })
    tasks['criticality_num'] = tasks['criticality'].map({'hard': 1, 'soft': 0})
    tasks['urgency'] = tasks['deadline'] / tasks['burst_time'].clip(1)
    tasks['urgency'] = tasks['urgency'].clip(0, 100)
    return tasks

# Core Logic
class TinyTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_heads=2, num_layers=2, output_dim=3):
        super(TinyTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True), num_layers
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        return self.output_fc(x)

def create_labels_and_features(tasks):
    """Process dataframe into features, labels, and clusters."""
    print("Starting feature processing...")
    scaler = MinMaxScaler()
    features = scaler.fit_transform(tasks[['burst_time', 'criticality_num', 'urgency']])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    print("Features scaled and converted to tensor.")

    labels = []
    for i, task in tasks.iterrows():
        if task['deadline'] < 50 and task['criticality'] == 'hard':
            labels.append(2)  # High priority
        elif task['burst_time'] < 40 and task['deadline'] < 120:
            labels.append(1)  # Medium priority
        else:
            labels.append(0)  # Low priority
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Cluster with GMM
    print("Starting GMM clustering...")
    gmm = GaussianMixture(n_components=3, covariance_type='tied', init_params='kmeans', random_state=42, max_iter=100)
    clusters = gmm.fit_predict(features)
    print("GMM clustering completed.")

    # Map clusters to heuristic labels
    cluster_to_label = {}
    for c in range(3):
        cluster_mask = clusters == c
        if cluster_mask.sum() > 0:
            most_common_label = pd.Series(labels)[cluster_mask].mode()[0]
            cluster_to_label[c] = most_common_label
    mapped_clusters = np.array([cluster_to_label[c] for c in clusters])

    # Validate clusters
    unique, counts = np.unique(mapped_clusters, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")
    if len(unique) > 1:
        silhouette = silhouette_score(features, mapped_clusters)
        print(f"Silhouette Score: {silhouette:.3f}")
    else:
        print("Silhouette Score not computed: only one cluster found")
        silhouette = -1

    # t-SNE visualization with sampling
    print("Starting t-SNE visualization...")
    sample_size = min(1000, len(tasks))  # Limit to 1000 points
    sample_idx = np.random.choice(len(tasks), sample_size, replace=False)
    try:
        tsne = TSNE(n_components=2, random_state=42, max_iter=300)  # Use max_iter for newer versions
        tsne_features = tsne.fit_transform(features[sample_idx])
    except TypeError as e:
        print(f"TSNE parameter error: {e}. Falling back to default iterations...")
        tsne = TSNE(n_components=2, random_state=42)  # Fallback for older versions
        tsne_features = tsne.fit_transform(features[sample_idx])
    tsne_labels = np.array(labels)[sample_idx]
    tsne_clusters = mapped_clusters[sample_idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=[['green', 'yellow', 'red'][l] for l in tsne_labels], alpha=0.5)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=[['green', 'yellow', 'red'][c] for c in tsne_clusters], marker='x', s=50, label='Clusters')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.title('t-SNE of Features with Clusters (Sampled)')
    plt.legend()
    plt.savefig(os.path.join(PROJECT_PATH, 'tsne_features.png'))
    plt.close()
    print("t-SNE visualization completed.")

    return features_tensor, torch.tensor(labels, dtype=torch.long), scaler, mapped_clusters

def run_training_stage(model, features, labels, epochs, stage_name, batch_size=128):
    """Run a training loop with validation and accuracy reporting."""
    print(f"\n--- Running {stage_name} ---")
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 2.0], device=device))  # Weight for 0, 1, 2
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    best_val_loss = float('inf')
    for epoch in range(epochs):
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        else:
            print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    # Evaluate accuracy on validation set
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f"  {stage_name} Validation Accuracy: {accuracy:.4f}")
    return model

# Main Execution
if __name__ == '__main__':
    PROJECT_PATH = r"D:\KLH\PROJECTS\SEMI SUPERVISED RTOS"
    if not os.path.exists(PROJECT_PATH):
        os.makedirs(PROJECT_PATH)
        print(f"Created directory: {PROJECT_PATH}")

    # Stage 1: Pre-training on Easy Data
    easy_tasks = generate_easy_data(num_tasks=2000)
    easy_features, easy_labels, _, _ = create_labels_and_features(easy_tasks)
    model = TinyTransformer(input_dim=3, output_dim=3).to(device)
    model = run_training_stage(model, easy_features, easy_labels, epochs=30, stage_name="Stage 1: Pre-training on Easy Data")
    print("✅ Model has learned the ideal patterns.")

    # Stage 2: Fine-tuning on Realistic Data
    print("\nGenerating realistic dataset for fine-tuning...")
    realistic_tasks_list = [generate_realistic_data(2000) for _ in range(20)]
    realistic_tasks = pd.concat(realistic_tasks_list, ignore_index=True)
    print(f"Total realistic data size: {len(realistic_tasks)} tasks")
    realistic_features, realistic_labels, scaler_for_saving, _ = create_labels_and_features(realistic_tasks)
    model = run_training_stage(model, realistic_features, realistic_labels, epochs=50, stage_name="Stage 2: Fine-tuning on Realistic Data")
    print("✅ Model has adapted to realistic, complex data.")

    # Save model and scaler
    model_path = os.path.join(PROJECT_PATH, 'SILVER_CS.pth')
    scaler_path = os.path.join(PROJECT_PATH, 'silver_cs_scaler.pkl')
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_for_saving, scaler_path)
    print(f"\n🏆 Final model saved to: '{model_path}'")
    print(f"✅ Scaler saved to: '{scaler_path}'")
    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB" if device.type == 'cuda' else "Running on CPU")