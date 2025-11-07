# Machine Learning Clustering Models Tutorial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Latest-blue.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sekertutku/Machine-Learning---Clustering-Models)

## Overview

This repository provides a comprehensive tutorial on **unsupervised machine learning clustering techniques** using Python. The project demonstrates 2 fundamental clustering algorithms with synthetic data generation, mathematical explanations, interactive visualizations using Plotly and Matplotlib, and comprehensive performance comparisons. Learn how to discover hidden patterns in unlabeled data!

## 🎮 Interactive Demo

**👉 [Run the Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models)**

*For the best experience with interactive Plotly visualizations and pre-configured environment, use the Kaggle notebook above. All models are ready to run with visual explanations and dendrograms!*

## Table of Contents

- [Introduction](#introduction)
- [What is Clustering?](#what-is-clustering)
- [Dataset](#dataset)
- [Clustering Algorithms](#clustering-algorithms)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Algorithm Comparison](#algorithm-comparison)
- [Mathematical Foundations](#mathematical-foundations)
- [Visualizations](#visualizations)
- [When to Use Which Algorithm](#when-to-use-which-algorithm)
- [Contributing](#contributing)
- [References](#references)

## Introduction

**Clustering** is an unsupervised learning technique that groups similar data points together without predefined labels. Unlike supervised learning (classification and regression), clustering discovers hidden patterns and structures in unlabeled data. This tutorial explores two fundamental clustering approaches that form the foundation of modern data analysis.

## What is Clustering?

### Supervised vs Unsupervised Learning

| Learning Type | Has Labels? | Examples | Goal |
|---------------|-------------|----------|------|
| **Supervised** | ✅ Yes (y) | Classification, Regression | Predict labels |
| **Unsupervised** | ❌ No labels | Clustering, Dimensionality Reduction | Discover patterns |

**Clustering Use Cases:**
- 🛒 **Customer Segmentation**: Group customers by behavior
- 🧬 **Gene Expression Analysis**: Identify gene patterns
- 📸 **Image Segmentation**: Group similar pixels
- 📄 **Document Clustering**: Organize similar documents
- 🔍 **Anomaly Detection**: Find unusual patterns
- 🎵 **Music Recommendation**: Group similar songs

## Dataset

### Synthetic Data Generation

This tutorial uses **synthetically generated data** to clearly demonstrate clustering concepts:

**Dataset Characteristics:**
- **Total Samples**: 3,000 points (K-Means) / 300 points (Hierarchical)
- **Clusters**: 3 distinct groups
- **Features**: 2-dimensional (x, y coordinates)
- **Distribution**: Normal (Gaussian) distribution

**Cluster Configuration:**

| Cluster | Location | Mean (x, y) | Std Dev | Points |
|---------|----------|-------------|---------|--------|
| 1 | Bottom-left | (25, 25) | 5 | 1,000 |
| 2 | Top-right | (55, 60) | 5 | 1,000 |
| 3 | Bottom-right | (55, 15) | 5 | 1,000 |

**Why Synthetic Data?**
- ✅ **Ground Truth Known**: We know the correct clusters
- ✅ **Clean & Controlled**: No missing values or noise
- ✅ **Reproducible**: Same results with random_state
- ✅ **Educational**: Perfect for learning concepts
- ✅ **Visual Clarity**: Easy to visualize in 2D

**Code Example:**
```python
# Cluster 1 (Bottom-left)
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

# Cluster 2 (Top-right)
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

# Cluster 3 (Bottom-right)
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)
```

## Clustering Algorithms

### 1. K-Means Clustering

**Concept**: Partitions data into **K distinct clusters** by minimizing within-cluster variance. The algorithm assigns each point to the nearest centroid and iteratively updates centroids until convergence.

#### How K-Means Works

**Algorithm Steps:**
1. **Initialize**: Choose K random centroids
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids (mean of assigned points)
4. **Repeat**: Steps 2-3 until convergence
5. **Output**: K clusters with final centroids

**Mathematical Formula:**
```
Minimize: Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

Where:
- `K` = Number of clusters
- `Ci` = Cluster i
- `μi` = Centroid of cluster i
- `||x - μi||` = Euclidean distance

#### Elbow Method

**Purpose**: Find optimal number of clusters (K)

**How it Works:**
- Plot K (1 to 15) vs WCSS (Within-Cluster Sum of Squares)
- Look for "elbow point" where WCSS decrease slows
- Elbow indicates optimal K

**WCSS Formula:**
```
WCSS = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

**Our Result:**
- **Elbow Point: K=3** ✅
- Matches ground truth perfectly!

#### Key Parameters

```python
KMeans(
    n_clusters=3,        # Number of clusters
    init='k-means++',    # Smart initialization (default)
    max_iter=300,        # Maximum iterations
    random_state=42      # For reproducibility
)
```

#### Advantages ✅

- ⚡ **Fast & Efficient**: O(n × K × iterations)
- 📊 **Scalable**: Works with large datasets (1000+ points per cluster)
- 🎯 **Simple**: Easy to implement and understand
- 📈 **Centroids**: Provides cluster centers for interpretation

#### Disadvantages ❌

- 🎲 **Must Specify K**: Need to know number of clusters beforehand
- 🔄 **Initialization Sensitive**: Different starts → different results
- ⭕ **Assumes Spherical**: Works best with round clusters
- 📉 **Outlier Sensitive**: Outliers affect centroid positions

#### Performance

**Dataset**: 3,000 points (1,000 per cluster)
**Processing Time**: Fast (< 1 second)
**Result**: Successfully identified 3 clusters with distinct centroids

### 2. Hierarchical Clustering

**Concept**: Builds a **hierarchy of clusters** without specifying K beforehand. Creates a tree-like structure (dendrogram) showing relationships between all data points.

#### Types of Hierarchical Clustering

**Agglomerative (Bottom-Up)** ⬆️ *[We use this]*
- Start: Each point is its own cluster
- Process: Merge closest clusters iteratively
- End: All points in one cluster

**Divisive (Top-Down)** ⬇️
- Start: All points in one cluster
- Process: Split clusters recursively
- End: Each point is its own cluster

#### Linkage Methods

| Method | Distance Calculation | Use Case |
|--------|---------------------|----------|
| **Ward** | Minimizes variance | Most common (we use this) |
| **Single** | Minimum distance | Long, thin clusters |
| **Complete** | Maximum distance | Compact clusters |
| **Average** | Average distance | Balanced approach |

**Ward Linkage Formula:**
```
d(Ci, Cj) = min(x ∈ Ci, y ∈ Cj) ||x - y||
```

#### Dendrogram

**Definition**: Tree diagram showing hierarchical relationships

**Components:**
- **Y-axis**: Distance/dissimilarity
- **X-axis**: Data points
- **Branches**: Cluster merges
- **Horizontal Line**: Cut to determine K

**How to Read:**
1. Start at bottom (individual points)
2. Follow branches upward (merges)
3. Draw horizontal line at desired distance
4. Number of vertical lines crossed = K clusters

**Our Result:**
- **Cut at distance ≈100 → K=3 clusters** ✅
- Dendrogram clearly shows 3 distinct groups!

#### Key Parameters

```python
AgglomerativeClustering(
    n_clusters=3,           # Number of clusters
    affinity='euclidean',   # Distance metric
    linkage='ward'          # Ward minimizes variance
)
```

#### Advantages ✅

- 🎯 **No K Required**: Use dendrogram to choose K
- 🌳 **Dendrogram**: Visual hierarchy of relationships
- 📏 **Any Distance Metric**: Flexible distance measures
- 🔗 **Captures Hierarchy**: Shows nested cluster structure

#### Disadvantages ❌

- 🐢 **Slow**: O(n³) computational complexity
- 💾 **Memory Intensive**: Stores full distance matrix
- 📊 **Small Datasets Only**: Not suitable for large data (< 5000 points)
- 🔒 **Irreversible**: Once merged, cannot undo
- 📉 **Noise Sensitive**: Outliers affect hierarchy

#### Performance

**Dataset**: 300 points (100 per cluster)
**Processing Time**: Slower (few seconds)
**Result**: Successfully identified 3 clusters with clear dendrogram

**Why Smaller Dataset?**
- Hierarchical is O(n³) → computationally expensive
- For educational clarity and reasonable runtime
- In production, use K-Means for large data

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

## Installation

### Option 1: Use Kaggle (Recommended) ⭐

The easiest way to explore this tutorial is on Kaggle where everything is pre-configured:

👉 **[Open Interactive Notebook on Kaggle](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models)**

### Option 2: Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/sekertutku/Machine-Learning---Clustering-Models.git
cd Machine-Learning---Clustering-Models
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the notebook:**
```bash
jupyter notebook machine-learning-clustering-models.ipynb
```

Or execute the Python script:
```bash
python machine-learning-clustering-models.py
```

## Usage

### Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate synthetic data
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)

x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))
data = pd.DataFrame({"x": x, "y": y})

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)
print(f"Cluster centers:\n{kmeans.cluster_centers_}")

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
h_clusters = hierarchical.fit_predict(data)

# Create Dendrogram
linkage_matrix = linkage(data, method='ward')
dendrogram(linkage_matrix)
plt.show()
```

### Elbow Method Example

```python
# Find optimal K
wcss = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 15), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

## Key Features

### 🎯 Comprehensive Algorithm Coverage
- **K-Means**: Fast partitioning with centroids
- **Hierarchical**: Dendrogram-based clustering
- **Elbow Method**: Optimal K selection
- **Ward Linkage**: Variance minimization

### 📊 Rich Visualizations
- **Ground Truth Plots**: See actual cluster distribution
- **Combined Data**: Visualize unsupervised challenge
- **Elbow Curves**: K optimization visualization
- **Dendrogram**: Hierarchical relationships
- **Cluster Results**: Color-coded final clusters
- **Interactive Plotly**: Hover, zoom, pan capabilities

### 📐 Mathematical Explanations
- **LaTeX Formulas**: Clear mathematical notation
- **WCSS Calculation**: Within-cluster sum of squares
- **Distance Metrics**: Euclidean distance explanation
- **Algorithm Steps**: Step-by-step breakdowns

### 🔬 Educational Content
- **Markdown Documentation**: Extensive theory
- **Pros & Cons**: For each algorithm
- **When to Use**: Practical guidance
- **Code Comments**: Line-by-line explanations

### ⚡ Performance Analysis
- **Dataset Sizes**: 3,000 vs 300 points comparison
- **Computational Complexity**: O(n) analysis
- **Speed Comparison**: K-Means vs Hierarchical
- **Scalability Discussion**: Large dataset recommendations

## Algorithm Comparison

### Performance Summary

| Feature | K-Means | Hierarchical |
|---------|---------|--------------|
| **Speed** | ⚡ Fast | 🐢 Slow |
| **Dataset Size** | Large (3,000 points) | Small (300 points) |
| **K Selection** | Must specify (use Elbow) | From dendrogram |
| **Scalability** | ✅ 10,000+ points | ⚠️ < 5,000 points |
| **Interpretability** | Good (centroids) | Excellent (dendrogram) |
| **Cluster Shape** | Spherical | Any shape |
| **Complexity** | O(n×K×iterations) | O(n³) |
| **Memory** | Low | High |
| **Visualization** | Centroids | Dendrogram tree |

### When to Use Each Algorithm

#### Use K-Means When: ✅

- ⚡ **Speed Priority**: Need fast results
- 📊 **Large Datasets**: 10,000+ data points
- 🎯 **Know Approximate K**: Have domain knowledge
- ⭕ **Spherical Clusters**: Round-shaped groups
- 💻 **Production Systems**: Real-time clustering
- 📈 **Scalability**: Dataset will grow

**Example Use Cases:**
- Customer segmentation (millions of customers)
- Image compression (large images)
- Real-time recommendation systems

#### Use Hierarchical When: 🌳

- 🎯 **Unknown K**: Don't know number of clusters
- 🔍 **Exploratory Analysis**: Discovering structure
- 📊 **Small/Medium Data**: < 5,000 points
- 🌳 **Need Hierarchy**: Relationships matter
- 📖 **Interpretability**: Need to explain results
- 🔬 **Research**: Detailed analysis required

**Example Use Cases:**
- Gene expression analysis (limited samples)
- Document clustering (research papers)
- Taxonomy creation (biological classification)

### Detailed Comparison

#### Dataset Size Impact

**K-Means:**
- ✅ Processed **3,000 points** efficiently
- ✅ Can handle 100,000+ points
- ✅ Linear scaling with data size

**Hierarchical:**
- ⚠️ Used **300 points** (computational constraint)
- ⚠️ Struggles with > 5,000 points
- ⚠️ Cubic scaling (O(n³))

#### Computational Complexity

**K-Means:**
```
Time: O(n × K × iterations)
Space: O(n + K)

n = 3,000 points
K = 3 clusters
iterations ≈ 10-50
→ Very fast!
```

**Hierarchical:**
```
Time: O(n³) or O(n²log n) with optimizations
Space: O(n²) for distance matrix

n = 300 points
→ Manageable
n = 3,000 points
→ Too slow!
```

#### Visualization Quality

**K-Means:**
- ⭐ Shows cluster centroids (yellow stars)
- ⭐ Color-coded clusters
- ⭐ Clear separation visible

**Hierarchical:**
- ⭐⭐ Dendrogram shows full hierarchy
- ⭐⭐ See all merge decisions
- ⭐⭐ Choose K by cutting tree

## Mathematical Foundations

### K-Means Mathematics

**Objective Function:**
```
Minimize J = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

**Where:**
- `J` = Total within-cluster variance
- `K` = Number of clusters
- `Ci` = Set of points in cluster i
- `μi` = Centroid of cluster i
- `||x - μi||` = Euclidean distance

**Centroid Update:**
```
μi = (1/|Ci|) × Σ(x ∈ Ci) x
```

**Euclidean Distance (2D):**
```
d(p, q) = √[(x₂-x₁)² + (y₂-y₁)²]
```

### Hierarchical Clustering Mathematics

**Ward Linkage Distance:**
```
d(Ci, Cj) = √[2×ni×nj/(ni+nj)] × ||μi - μj||
```

**Where:**
- `ni`, `nj` = Number of points in clusters i and j
- `μi`, `μj` = Centroids of clusters i and j

**Dendrogram Height:**
```
Height = Distance at which clusters merge
```

**Optimal K Selection:**
```
Cut dendrogram at height where:
- Large vertical gaps appear
- Significant distance increase
```

## Visualizations

### Visualization Gallery

#### 1. Ground Truth (Separated Clusters)
Shows the 3 clusters in different colors before combining - our "answer key"

#### 2. Combined Data (Unsupervised Challenge)
All points in gray - what the algorithm sees (no labels!)

#### 3. Elbow Curve
Plot of K vs WCSS showing the "elbow" at K=3

**Key Features:**
- Marker points at each K value
- Red vertical line at optimal K=3
- Grid for easy reading
- Clear axis labels

#### 4. K-Means Results
Final clustering with:
- 3 color-coded clusters (red, green, blue)
- Yellow star centroids (⭐)
- Larger figure size (12×8)
- Legend and grid

#### 5. Dendrogram
Hierarchical tree showing:
- Merge sequence (bottom to top)
- Distance scale (y-axis)
- Red horizontal cut line (K=3)
- All 300 data points (x-axis)

#### 6. Hierarchical Results
Final clustering matching dendrogram cut:
- Same 3-cluster structure
- Color-coded groups
- No centroids (not applicable)

### Interactive Features (Plotly)

While the provided code uses Matplotlib, Plotly integration enables:
- 🖱️ **Hover**: See exact coordinates
- 🔍 **Zoom**: Focus on specific regions
- 📸 **Pan**: Move around plot
- 💾 **Export**: Save as images

## Key Insights

### Critical Findings

**✅ Both Algorithms Succeeded:**
- K-Means correctly identified 3 clusters (3,000 points)
- Hierarchical also found 3 clusters (300 points)
- Elbow Method confirmed K=3
- Dendrogram clearly showed 3-cluster structure

**📊 Performance Insights:**
- **K-Means**: 10x more data processed efficiently
- **Hierarchical**: Detailed hierarchy with smaller dataset
- **Speed**: K-Means much faster for large data
- **Interpretability**: Dendrogram superior for exploration

**🎯 Optimal Use Cases:**
- **Production**: Use K-Means for speed and scale
- **Analysis**: Use Hierarchical for understanding
- **Combined**: Hierarchical exploration → K-Means production

### Best Practices

**Data Preparation:**
- ✅ Scale features (especially for distance-based algorithms)
- ✅ Handle missing values
- ✅ Remove outliers (or use robust methods)
- ✅ Check for high dimensionality (reduce if needed)

**K Selection:**
- ✅ Use Elbow Method for K-Means
- ✅ Use Dendrogram for Hierarchical
- ✅ Consider domain knowledge
- ✅ Try multiple K values

**Evaluation:**
- ✅ Silhouette Score (cluster quality)
- ✅ Davies-Bouldin Index (separation)
- ✅ Visual inspection (always!)
- ✅ Domain expert validation

**Performance:**
- ✅ Start small (sample data for testing)
- ✅ Use K-Means for > 5,000 points
- ✅ Consider Mini-Batch K-Means for huge datasets
- ✅ Use random_state for reproducibility

### Common Pitfalls

❌ **Using Hierarchical on Large Data**
- O(n³) complexity makes it impractical
- Use K-Means or Mini-Batch K-Means instead

❌ **Not Scaling Features**
- Distance-based algorithms need normalized features
- Different scales → biased clustering

❌ **Ignoring Domain Knowledge**
- Always validate clusters make sense
- Not all mathematically optimal clusters are meaningful

❌ **Choosing K Randomly**
- Use Elbow Method or Dendrogram
- Consider business/domain requirements

❌ **Expecting Perfect Clusters**
- Real data is messy
- Overlapping clusters are common
- Algorithms make best effort

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add DBSCAN clustering algorithm
- Implement Silhouette Score analysis
- Add real-world datasets
- Create interactive Plotly visualizations
- Add cross-validation techniques
- Improve documentation
- Add unit tests

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## References

### Course
- **Udemy**: MACHINE LEARNING by DATAI TEAM

### Documentation
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Hierarchical Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [SciPy Dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Plotly Python](https://plotly.com/python/)

### Algorithms & Theory
- [K-Means Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)
- [Hierarchical Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)
- [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- [Ward Linkage](https://en.wikipedia.org/wiki/Ward%27s_method)
- [Dendrogram](https://en.wikipedia.org/wiki/Dendrogram)

### Related Projects

**My Machine Learning Series:**

- 🔍 **Clustering Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-clustering-models) [[GitHub]](https://github.com/sekertutku/Machine-Learning---Clustering-Models) *(Current)*

- 🎯 **Classification Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-classifications-models) [[GitHub]](https://github.com/sekertutku/Machine-Learning---Classifications-Models)

- 📈 **Regression Models** - [[Kaggle]](https://www.kaggle.com/code/dandrandandran2093/machine-learning-regression-models) [[GitHub]](https://github.com/sekertutku/Machine-Learning---Regression-Models)

## Acknowledgments

Special thanks to:
- **DATAI TEAM** for the comprehensive machine learning course
- **Scikit-learn developers** for excellent clustering implementations
- **SciPy team** for hierarchical clustering tools
- **Matplotlib & Plotly** teams for visualization libraries
- **The open-source community** for making machine learning accessible

---

**Note**: This tutorial is intended for educational purposes. The synthetic data and algorithms demonstrate fundamental clustering concepts. For production systems, always validate on real data and consider additional factors like scalability, interpretability, and domain requirements.

## 📞 Connect

If you have questions or suggestions:
- Open an issue in this repository
- Connect on [Kaggle](https://www.kaggle.com/dandrandandran2093)
- Visit my website: [tutkufurkan.com](https://www.tutkufurkan.com/)
- Star this repository if you found it helpful! ⭐

---

**Happy Clustering! 🎯🔍✨**

🌐 More projects at [tutkufurkan.com](https://www.tutkufurkan.com/)
