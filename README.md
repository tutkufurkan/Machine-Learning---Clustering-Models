# Machine Learning Clustering Models Tutorial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Latest-blue.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sekertutku/Machine-Learning---Clustering-Models)

## Overview

A comprehensive tutorial on **unsupervised machine learning clustering techniques** using Python. Learn K-Means and Hierarchical Clustering with synthetic data, mathematical explanations, interactive visualizations, and detailed performance comparisons.

## 🎮 Interactive Demo

**👉 [Run the Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models)**

## Table of Contents

- [What is Clustering?](#what-is-clustering)
- [Clustering Algorithms](#clustering-algorithms)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Comparison](#algorithm-comparison)
- [Key Insights](#key-insights)
- [References](#references)

## What is Clustering?

**Clustering** is an unsupervised learning technique that groups similar data points together without predefined labels. Unlike supervised learning, clustering discovers hidden patterns in unlabeled data.

### Supervised vs Unsupervised Learning

| Type | Has Labels? | Examples | Goal |
|------|-------------|----------|------|
| **Supervised** | ✅ Yes | Classification, Regression | Predict labels |
| **Unsupervised** | ❌ No | Clustering | Discover patterns |

**Common Use Cases:**
- 🛒 Customer segmentation
- 🧬 Gene expression analysis
- 📸 Image segmentation
- 📄 Document clustering
- 🔍 Anomaly detection

## Clustering Algorithms

### 1. K-Means Clustering

**Concept**: Partitions data into K clusters by minimizing within-cluster variance.

**Algorithm:**
1. Choose K (number of clusters)
2. Initialize K random centroids
3. Assign points to nearest centroid
4. Update centroids (mean of assigned points)
5. Repeat until convergence

**Formula:**
$$\text{Minimize: } \sum_{i=1}^{K}\sum_{x \in C_i}||x - \mu_i||^2$$

**Elbow Method**: Plot K vs WCSS to find optimal number of clusters. Look for the "elbow point" where WCSS decrease slows down.

**Advantages:**
- ⚡ Fast and efficient
- 📊 Scalable to large datasets
- 🎯 Simple to implement

**Disadvantages:**
- 🎲 Must specify K beforehand
- 🔄 Sensitive to initialization
- ⭕ Assumes spherical clusters

### 2. Hierarchical Clustering

**Concept**: Builds a hierarchy of clusters without specifying K beforehand. Creates a dendrogram (tree structure) showing relationships.

**Algorithm (Agglomerative):**
1. Start with each point as its own cluster
2. Merge two closest clusters
3. Repeat until one cluster remains
4. Cut dendrogram at desired height to get K clusters

**Formula:**
$$\text{Distance: } d(C_i, C_j) = \min_{x \in C_i, y \in C_j} ||x - y||$$

**Linkage Methods:**
- **Ward**: Minimizes variance (most common)
- **Single**: Minimum distance
- **Complete**: Maximum distance
- **Average**: Average distance

**Advantages:**
- 🌳 No need to specify K
- 📊 Dendrogram visualization
- 🔗 Captures hierarchical relationships

**Disadvantages:**
- 🐢 Slow (O(n³) complexity)
- 💾 Not suitable for large datasets
- 🔒 Merge decisions are irreversible

## Dataset

**Synthetic Data Generation**: 3 clusters with Gaussian distribution

| Cluster | Location | Mean (x, y) | Points (K-Means) | Points (Hierarchical) |
|---------|----------|-------------|------------------|-----------------------|
| 1 | Bottom-left | (25, 25) | 1,000 | 100 |
| 2 | Top-right | (55, 60) | 1,000 | 100 |
| 3 | Bottom-right | (55, 15) | 1,000 | 100 |

**Total**: 3,000 points for K-Means / 300 points for Hierarchical

**Why different sizes?** Hierarchical is computationally expensive (O(n³)), so we use a smaller dataset for reasonable runtime.

## Installation

### Option 1: Kaggle (Recommended) ⭐

👉 **[Open on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models)** - Everything pre-configured!

### Option 2: Local

```bash
# Clone repository
git clone https://github.com/sekertutku/Machine-Learning---Clustering-Models.git
cd Machine-Learning---Clustering-Models

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook machine-learning-clustering-models.ipynb
```

## Usage

### Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate data
x = np.concatenate([np.random.normal(25, 5, 1000), 
                    np.random.normal(55, 5, 1000),
                    np.random.normal(55, 5, 1000)])
y = np.concatenate([np.random.normal(25, 5, 1000),
                    np.random.normal(60, 5, 1000),
                    np.random.normal(15, 5, 1000)])
data = pd.DataFrame({"x": x, "y": y})

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)
print(f"Centroids:\n{kmeans.cluster_centers_}")

# Hierarchical
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
h_clusters = hierarchical.fit_predict(data)

# Dendrogram
linkage_matrix = linkage(data, method='ward')
dendrogram(linkage_matrix)
plt.show()
```

### Elbow Method

```python
# Find optimal K
wcss = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot
plt.plot(range(1, 15), wcss, marker='o')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

## Algorithm Comparison

### Performance Summary

| Feature | K-Means | Hierarchical |
|---------|---------|--------------|
| **Speed** | ⚡ Fast | 🐢 Slow |
| **Dataset Size** | Large (3,000 points) | Small (300 points) |
| **K Selection** | Must specify (Elbow Method) | From dendrogram |
| **Scalability** | ✅ 10,000+ points | ⚠️ < 5,000 points |
| **Visualization** | Centroids | Dendrogram tree |
| **Complexity** | O(n×K×iterations) | O(n³) |
| **Cluster Shape** | Spherical | Any shape |

### When to Use

**K-Means:**
- ✅ Large datasets (10,000+ points)
- ✅ Speed is critical
- ✅ Production systems
- ✅ Spherical clusters expected

**Hierarchical:**
- ✅ Unknown number of clusters
- ✅ Small/medium datasets (< 5,000 points)
- ✅ Need to visualize hierarchy
- ✅ Exploratory analysis

## Key Insights

**✅ Both Algorithms Succeeded:**
- K-Means: 3,000 points processed efficiently
- Hierarchical: 300 points with clear dendrogram
- Elbow Method confirmed K=3
- Dendrogram showed 3-cluster structure

**📊 Best Practices:**
- Use Elbow Method for K-Means optimization
- Use Dendrogram for Hierarchical K selection
- Scale features before clustering
- Start with K-Means for large data
- Use Hierarchical for exploratory analysis

**⚠️ Common Pitfalls:**
- Using Hierarchical on large datasets (too slow!)
- Not scaling features (distance-based algorithms need it)
- Choosing K randomly (use Elbow/Dendrogram)
- Ignoring domain knowledge

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
scipy>=1.11.0
jupyter>=1.0.0
```

## Contributing

Contributions welcome! Please open an issue first to discuss major changes.

**Ideas:**
- Add DBSCAN algorithm
- Implement Silhouette Score
- Add real-world datasets
- Create interactive Plotly visualizations

## License

Apache License 2.0 - see LICENSE file for details.

## References

### Course
- **Udemy**: MACHINE LEARNING by DATAI TEAM

### Documentation
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Hierarchical Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [SciPy Dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)

**My Machine Learning Series:**

- 🔍 **Clustering Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Clustering-Models) *(Current)*

- 🚀 **Advanced Topics** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-advanced-topics) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Advanced-Topics)

- 🎯 **Classification Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Classifications-Models)

- 📈 **Regression Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) [[GitHub]](https://github.com/tutkufurkan/Machine-Learning---Regression-Models)

## Acknowledgments

- DATAI TEAM for the machine learning course
- Scikit-learn and SciPy developers
- Open-source community

---

## 📞 Connect

- Open an issue for questions
- Connect on [Kaggle](https://www.kaggle.com/dandrandandran2093)
- Visit [tutkufurkan.com](https://www.tutkufurkan.com/)
- Star ⭐ if helpful!

---

**Happy Clustering! 🎯🔍**

🌐 [tutkufurkan.com](https://www.tutkufurkan.com/)
