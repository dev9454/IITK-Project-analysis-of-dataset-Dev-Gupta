import pandas as pd

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load the dataset into a DataFrame
wine_data = pd.read_csv(url, sep=';')
# Check for missing values
missing_values = wine_data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

#QUESTION 2:
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

# Bar chart of mean alcohol content by wine quality
mean_alcohol_by_quality = wine_data.groupby('quality')['alcohol'].mean()
quality_labels = mean_alcohol_by_quality.index

plt.figure(figsize=(8, 5))
plt.bar(quality_labels, mean_alcohol_by_quality, color='skyblue')
plt.title('Mean Alcohol Content by Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Mean Alcohol Content')
plt.xticks(quality_labels)
plt.show()

wine_quality_counts = wine_data['quality'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(wine_quality_counts.index, wine_quality_counts.values, color='lightcoral')
plt.title('Wine Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xticks(wine_quality_counts.index)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Select three continuous features for the 3D plot
feature1 = 'alcohol'
feature2 = 'citric acid'
feature3 = 'density'

#3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = wine_data[feature1]
y = wine_data[feature2]
z = wine_data[feature3]
c = wine_data['quality']  

scatter = ax.scatter(x, y, z, c=c, cmap='coolwarm')

ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel(feature3)
ax.set_title(f'3D Scatter Plot of {feature1}, {feature2}, and {feature3} by Wine Quality')

# Add a colorbar to indicate wine quality
cbar = plt.colorbar(scatter)
cbar.set_label('Quality')

plt.show()
#QUESTION 3
from sklearn.ensemble import RandomForestClassifier

# Separate features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False, inplace=True)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, orient='h')
plt.title('Feature Importance for Wine Quality')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

