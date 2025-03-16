import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the iris.csv file
df = pd.read_csv('iris1.csv')

# Display the first few rows of the dataset
print("Dataset:\n", df.head())

# Step 2: Standardize the data
# Assuming the columns are named: sepal_length, sepal_width, petal_length, petal_width, species
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
x = df.loc[:, features].values  # Features
y = df.loc[:, 'variety'].values  # Target (species)

# Standardize the features
x = StandardScaler().fit_transform(x)

# Step 3: Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
principal_df['variety'] = y

# Display the first few rows of the principal components
print("Principal Components:\n", principal_df.head())

# Step 4: Visualize the 2D projection
plt.figure(figsize=(8, 6))
targets = df['variety'].unique()  # Get unique species names
colors = ['r', 'g', 'b']  # Colors for each species

for target, color in zip(targets, colors):
    indices = principal_df['variety'] == target
    plt.scatter(principal_df.loc[indices, 'PC1'],
                principal_df.loc[indices, 'PC2'],
                c=color, label=target)

plt.title('2D PCA of Iris Dataset')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend()


# Step 5: Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio (PC1, PC2):", explained_variance)
print("Total variance explained by PC1 and PC2:", sum(explained_variance))
plt.show()