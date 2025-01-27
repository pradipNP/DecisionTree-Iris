# DECISION TREE IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : PRADEEP KUMAR KOHAR

*INTERN ID* : CODHC29

*DOMAIN* : MACHINE LEARNING

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

The code in your DecisionTree-Iris project is designed to classify Iris flower species (Setosa, Versicolor, Virginica) using a Decision Tree algorithm implemented in Python with the help of scikit-learn. Below is a detailed breakdown of the code and what each section does:

1. Importing Necessary Libraries
   - pandas: For data manipulation and analysis.
   - numpy: For numerical operations.
   - sklearn modules:
      - datasets: To load the Iris dataset.
      - train_test_split: For splitting the dataset into training and testing sets.
      - DecisionTreeClassifier: For creating the decision tree model.
      - export_text and plot_tree: For visualizing the decision tree.
Purpose: To ensure all tools are available to load data, build models, and visualize results.

2. Loading the Dataset
   from sklearn.datasets import load_iris
   iris = load_iris()
- The Iris dataset is a built-in dataset in scikit-learn. It contains 150 samples of Iris flowers with four features:
   - Sepal length
   - Sepal width
   - Petal length
   - Petal width
- The dataset also includes three target labels corresponding to the flower species:
   - Setosa
   - Versicolor
   - Virginica
- Purpose: To load and explore the data for classification.

3. Splitting the Data
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

- The dataset is split into training and testing sets:
   - 80% for training (X_train, y_train).
   - 20% for testing (X_test, y_test).
Purpose: To train the model on one portion of the data and evaluate it on another portion to ensure generalization.

4. Creating and Training the Decision Tree
   from sklearn.tree import DecisionTreeClassifier
   clf = DecisionTreeClassifier(max_depth=3, random_state=42)
   clf.fit(X_train, y_train)
- A DecisionTreeClassifier is created with a maximum depth of 3 to avoid overfitting.
- The model is trained using the fit method with the training data (X_train, y_train).
Purpose: To build a model that learns how to classify Iris species based on input features.

5. Evaluating the Model
   accuracy = clf.score(X_test, y_test)
   print(f"Accuracy: {accuracy * 100:.2f}%")
- The score method evaluates the model's performance on the test data.
- The accuracy percentage is printed to show how well the model predicts unseen data.
Purpose: To measure the model's effectiveness in classifying Iris species.

7. Visualizing the Decision Tree
   from sklearn.tree import plot_tree
   import matplotlib.pyplot as plt

   plt.figure(figsize=(12,8))
   plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
   plt.show()
- The plot_tree function generates a graphical representation of the decision tree:
   - feature_names: Labels the features (e.g., sepal length, petal width).
   - class_names: Labels the target classes (e.g., Setosa, Versicolor).
   - filled=True: Colors the nodes to indicate classification outcomes.
Purpose: To visually understand the decision-making process of the model.

7. Exporting the Decision Tree Rules
   from sklearn.tree import export_text
   tree_rules = export_text(clf, feature_names=iris.feature_names)
   print(tree_rules)
- The export_text function converts the decision tree into a text-based format.
- It shows the rules (conditions and thresholds) used at each decision node.
Purpose: To make the model's logic transparent and interpretable.

8. Saving the Visualization and Results
- You can save the decision tree plot and rules into files for documentation or sharing:
   plt.savefig("decision_tree_visualization.png")
   with open("decision_tree_rules.txt", "w") as f:
        f.write(tree_rules)
