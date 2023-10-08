import pandas as pd

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load the dataset into a DataFrame
wine_data = pd.read_csv(url, sep=';')
# Check for missing values
missing_values = wine_data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of wine quality scores
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=wine_data)
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Calculate summary statistics
summary_stats = wine_data.describe()

# Correlation matrix to explore relationships between features and quality
correlation_matrix = wine_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Box plots of features by wine quality
plt.figure(figsize=(12, 6))
sns.boxplot(x='quality', y='alcohol', data=wine_data)
plt.title('Alcohol Content by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Alcohol Content')
plt.show()

# Pair plot of features with wine quality
sns.pairplot(wine_data, hue='quality', palette='coolwarm')
plt.suptitle('Pair Plot of Features with Wine Quality', y=1.02)
plt.show()

# Violin plots of features by wine quality
plt.figure(figsize=(12, 6))
sns.violinplot(x='quality', y='sulphates', data=wine_data, inner='quart')
plt.title('Sulphates Distribution by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Sulphates')
plt.show()

# Scatter plot of alcohol content vs. citric acid
plt.figure(figsize=(8, 5))
sns.scatterplot(x='alcohol', y='citric acid', data=wine_data, hue='quality', palette='coolwarm')
plt.title('Scatter Plot of Alcohol Content vs. Citric Acid')
plt.xlabel('Alcohol Content')
plt.ylabel('Citric Acid')
plt.show()
