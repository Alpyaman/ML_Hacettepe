# Implementing k-Nearest Neighbors
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN Classifier

        Parameters
        ----------
        k : number of neighbors to consider
        distance_metric : 'euclidean' or 'manhattan'

        """
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))  
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
        
    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric. Please choose 'euclidean' or 'manhattan'.")
    
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def evaluate(self, X, y): 
        y_prediction = self.predict(X)
        accuracy = np.sum(y_prediction == y) / len(y)
        return accuracy
    
# Example data for the knn implementation to work
iris = load_iris() 
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Evaluate different k values and distance metrics
k_values = [1, 3, 5, 7, 9]
metrics = ['euclidean', 'manhattan']

results = {metric: [] for metric in metrics}

for metric in metrics:
    for k in k_values:
        knn = KNNClassifier(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        accuracy = knn.evaluate(X_test, y_test)  
        results[metric].append(accuracy)
        print(f"Metric: {metric}, k: {k}, Accuracy: {accuracy:.4f}")
    
# Plot results
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(k_values, results[metric], marker='o', label=metric)
plt.title('Accuracy vs k for different distance metrics')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid()    
plt.show()
    
# Evaluate best model
best_k = k_values[np.argmax(results['euclidean'])]
knn = KNNClassifier(k=best_k, distance_metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))