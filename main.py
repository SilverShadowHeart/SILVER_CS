import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.stats import norm
import seaborn as sns
from sklearn.manifold import TSNE

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set up device for GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Generate synthetic task data
def generate_task_data(num_tasks=1000):
    np.random.seed(42)
    criticalities = np.random.choice(['hard', 'soft'], num_tasks, p=[0.3, 0.7])
    burst_times = np.where(
        criticalities == 'hard',
        np.random.exponential(scale=5, size=num_tasks).clip(1, 10),  # Tight range
        np.random.exponential(scale=100, size=num_tasks).clip(50, 200)  # Wide range
    )
    deadlines = burst_times + np.random.randint(50, 150, num_tasks)  # Wider deadline range
    tasks = pd.DataFrame({
        'task_id': range(num_tasks),
        'burst_time': burst_times,
        'criticality': criticalities,
        'deadline': deadlines
    })
    tasks['criticality_num'] = tasks['criticality'].map({'hard': 1, 'soft': 0})
    tasks['urgency'] = tasks['deadline'] / tasks['burst_time'].clip(1)
    tasks['urgency'] = tasks['urgency'].clip(0, 100)  # Avoid extreme values
    corr = tasks[['burst_time', 'criticality_num', 'urgency']].corr()
    print("Feature correlations:\n", corr)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('feature_correlations.png')
    plt.close()

    # Plot feature distributions
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(tasks['burst_time'], bins=50)
    plt.title('Burst Time Distribution')
    plt.subplot(1, 3, 2)
    plt.hist(tasks['criticality_num'], bins=2)
    plt.title('Criticality Distribution')
    plt.subplot(1, 3, 3)
    plt.hist(tasks['urgency'], bins=50)
    plt.title('Urgency Distribution')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

    # Compute feature overlap (Bhattacharyya distance for burst_time)
    hard_tasks = tasks[tasks['criticality'] == 'hard']['burst_time']
    soft_tasks = tasks[tasks['criticality'] == 'soft']['burst_time']
    mu1, sigma1 = norm.fit(hard_tasks)
    mu2, sigma2 = norm.fit(soft_tasks)
    bc_distance = 0.25 * np.log(0.25 * (sigma1**2 / sigma2**2 + sigma2**2 / sigma1**2 + 2)) + \
                  0.25 * ((mu1 - mu2)**2 / (sigma1**2 + sigma2**2))
    print(f"Bhattacharyya distance (burst_time, hard vs soft): {bc_distance:.4f}")

    return tasks

# Cluster and label tasks
def cluster_and_label(tasks, batch_size=64):
    try:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(tasks[['burst_time', 'criticality_num', 'urgency']])
        features = torch.tensor(features, dtype=torch.float32)
        print(f"Features shape: {features.shape}")

        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError("Features contain NaN or infinite values")

        # Heuristic labels
        labels = []
        for i, task in tasks.iterrows():
            if task['deadline'] < 60 and task['criticality'] == 'hard':
                labels.append(2)  # High priority
            elif task['burst_time'] < 50:
                labels.append(1)  # Medium priority
            else:
                labels.append(0)  # Low priority

        # Cluster using K-means with heuristic labels as initialization
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(features)

        unique, counts = np.unique(clusters, return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique, counts))}")
        centroids = np.array([features[clusters == i].mean(axis=0) for i in unique])
        print(f"Cluster centroids: {centroids}")
        inter_cluster_distances = cdist(centroids, centroids, metric='euclidean')
        print(f"Inter-cluster distances:\n{inter_cluster_distances}")

        if len(np.unique(clusters)) > 1:
            silhouette = silhouette_score(features, clusters)
            print(f"Silhouette Score: {silhouette:.3f}")
        else:
            print("Silhouette Score not computed: only one cluster found")
            silhouette = -1

        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(features)
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=[['green', 'yellow', 'red'][l] for l in labels], alpha=0.5)
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=[['green', 'yellow', 'red'][c] for c in clusters], marker='x', s=50, label='Clusters')
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.title('t-SNE of Features with Clusters')
        plt.legend()
        plt.savefig('tsne_features.png')
        plt.close()

        tasks['cluster'] = clusters
        tasks['priority'] = labels
        tasks['color'] = tasks['priority'].map({0: 'green', 1: 'yellow', 2: 'red'})
        return tasks, features, labels
    except Exception as e:
        print(f"Error in cluster_and_label: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

# Tiny Transformer for priority prediction
class TinyTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, num_heads=2, num_layers=1):
        super(TinyTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True), num_layers
        )
        self.output_fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        return self.output_fc(x)

# Train and simulate scheduler
def train_and_simulate(tasks, features, labels, batch_size=64):
    try:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(features, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        model = TinyTransformer(input_dim=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch in dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Transformer Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

        model.eval()
        priorities = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size].to(device)
                outputs = model(batch_features)
                priorities.extend(outputs.argmax(dim=1).cpu().numpy())

        tasks['predicted_priority'] = priorities
        tasks['predicted_color'] = tasks['predicted_priority'].map({0: 'green', 1: 'yellow', 2: 'red'})
        return tasks, model
    except Exception as e:
        print(f"Error in train_and_simulate: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

# Visualize tasks
def visualize_tasks(tasks):
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(tasks['burst_time'], tasks['deadline'], c=tasks['predicted_color'], alpha=0.5)
        plt.xlabel('Burst Time (ms)')
        plt.ylabel('Deadline (ms)')
        plt.title('Task Distribution with Predicted Priority')

        plt.subplot(1, 2, 2)
        plt.scatter(tasks['burst_time'], tasks['urgency'], c=tasks['color'], alpha=0.5)
        plt.xlabel('Burst Time (ms)')
        plt.ylabel('Urgency')
        plt.title('Task Distribution with Heuristic Priority')
        
        plt.tight_layout()
        plt.savefig('task_distribution.png')
        plt.show()
    except Exception as e:
        print(f"Error in visualize_tasks: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

# Main
if __name__ == '__main__':
    try:
        tasks = generate_task_data(1000)
        tasks, features, labels = cluster_and_label(tasks, batch_size=64)
        tasks, model = train_and_simulate(tasks, features, labels, batch_size=64)
        visualize_tasks(tasks)
        print(tasks[['task_id', 'burst_time', 'criticality', 'deadline', 'cluster', 'priority', 'predicted_priority', 'predicted_color']].head())
        if device.type == 'cuda':
            print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        torch.cuda.empty_cache()