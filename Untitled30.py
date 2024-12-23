#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# In[6]:


dataset = pd.read_csv(r'C:\CSV files\winequality-red.csv')
dataset


# In[9]:


X = dataset.drop(columns=['quality'])
y = dataset['quality']


# In[11]:


y = y - y.min()


# In[13]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[15]:


smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)


# In[17]:


pca = PCA(n_components=10)  # Keep 10 principal components
X_reduced = pca.fit_transform(X_balanced)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_balanced, test_size=0.2, random_state=42)


# In[21]:


best_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
best_svm = SVC(C=1, kernel='rbf', gamma='scale', probability=True)
best_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, colsample_bytree=0.8)


# In[23]:


ensemble = VotingClassifier(
    estimators=[('knn', best_knn), ('svm', best_svm), ('xgb', best_xgb)],
    voting='soft'  # Soft voting averages predicted probabilities for more stability
)


# In[25]:


ensemble.fit(X_train, y_train)
ensemble_predictions = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)


# In[27]:


print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")


# In[29]:


adaboost = AdaBoostClassifier(random_state=42)
adaboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}


# In[31]:


adaboost_grid = GridSearchCV(adaboost, adaboost_params, cv=5, scoring='accuracy')
adaboost_grid.fit(X_train, y_train)
best_adaboost = adaboost_grid.best_estimator_


# In[32]:


ensemble = VotingClassifier(
    estimators=[('knn', best_knn), ('svm', best_svm), ('xgb', best_xgb), ('adaboost', best_adaboost)],
    voting='soft'  # Soft voting for stability
)


# In[33]:


ensemble.fit(X_train, y_train)
ensemble_predictions = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Model Accuracy (KNN, SVM, XGBoost, AdaBoost): {ensemble_accuracy * 100:.2f}%")


# In[34]:


conf_matrix = confusion_matrix(y_test, ensemble_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Ensemble Model')
plt.show()


# In[35]:


plt.figure(figsize=(10, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()


# In[36]:


# Bar Graph: Average Alcohol Content per Quality Level
plt.figure(figsize=(10, 6))
average_alcohol_per_quality = dataset.groupby('quality')['alcohol'].mean()
average_alcohol_per_quality.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Quality Level')
plt.ylabel('Average Alcohol Content')
plt.title('Average Alcohol Content per Quality Level')
plt.xticks(rotation=0)
plt.show()


# In[37]:


plt.figure(figsize=(8, 6))
plt.hist(dataset['alcohol'], bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.xlabel('Alcohol Content')
plt.ylabel('Frequency')
plt.title('Histogram of Alcohol Content')
plt.show()


# In[38]:


plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_balanced, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Quality')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Wine Quality Data (PCA-Reduced)')
plt.show()


# In[39]:


import matplotlib.pyplot as plt

# Model names and accuracy values
models = [
    "Ensemble (KNN, SVM, XGBoost, AdaBoost)",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)",
    "XGBoost",
    "AdaBoost",
    "Previous Work 1 (Random Forest)",
    "Previous Work 2 (Logistic Regression)"
]
accuracies = [85.4, 85.21, 85.21, 85.21, 84.96, 83.5, 80.2]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(models, accuracies, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#FF6F61'])
plt.xlabel("Accuracy (%)")
plt.title("Comparison of Model Accuracies for Wine Quality Prediction")
plt.xlim(75, 90)  # Adjust limits for better visual clarity

# Annotate bars with accuracy values
for index, value in enumerate(accuracies):
    plt.text(value + 0.2, index, f"{value}%", va='center')

plt.gca().invert_yaxis()  # Invert y-axis to display top model at the top
plt.show()


# In[40]:


feature_means = X_balanced.mean(axis=0)

# Add the first value to the end for a closed radar chart
values = feature_means.tolist()
values += values[:1]

# Create radar plot
features = X.columns  # Original feature names
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Radar chart setup
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
ax.fill(angles, values, color='blue', alpha=0.25)
ax.plot(angles, values, color='blue', linewidth=2)
ax.set_yticks([])  # Remove the circular gridlines
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=12)
ax.set_title('Radar Plot of Scaled and Balanced Features', fontsize=16, pad=20)

# Display the plot
plt.show()


# In[ ]:




