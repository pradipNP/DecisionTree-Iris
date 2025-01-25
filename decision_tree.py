import numpy as np # For numerical operations
import pandas as pd # For working with tabular data
import matplotlib.pyplot as plt # For basic plotting 
import seaborn as sns # For advanced visualizations

# Import scikit-learn modules
from sklearn.datasets import load_iris # To load the Iris dataset
from sklearn.model_selection import train_test_split # To split the dataset
from sklearn.tree import DecisionTreeClassifier, plot_tree # For building and visualizing the decision tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # For evaluating the model

# Load the Iris dataset
data = load_iris()

# Create a DataFrame for easy data handling
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target #Add the target (flower type) column

# Print the first few rows to understand the structure
print(df.head())

# Print dataset description (optional)
print(data.DESCR)

# Check the dataset shape and data types
print("Shape of dataset:", df.shape)
print(df.info())

# Check for missing values
print("Missing values in each column:\n", df.isnull())

# Visualize the dataset
sns.pairplot(df, hue='target', palette='Set2') # Pairplot showing relationships between features
plt.show()

# Separate features (X) and target (y)
X = df.drop('target', axis=1) # Features: All column except 'target
y = df['target'] # Target: The 'target' column

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(randomo_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training complete!")

# Visualize the tree
plt.figure(figsize=(15,10)) # Adjust size
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

#Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

