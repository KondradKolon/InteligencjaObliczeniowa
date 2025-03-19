# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv("iris1.csv")
print("Original DataFrame:")
print(df)

# Splitting the dataset into training (70%) and testing (30%) sets with a random seed of 13
(train_set, test_set) = train_test_split(
    df.values, train_size=0.7, random_state=295132)

# Checking which records are in the test set and how many records it contains
print("\nTest set:")
print(test_set)
print("Number of records in the test set:", test_set.shape[0])

# Splitting the training and testing sets into inputs (numeric columns) and classes (iris species)
train_inputs = train_set[:, 0:4]  # First 4 columns are the features (inputs)
train_classes = train_set[:, 4]   # Last column is the target (classes)
test_inputs = test_set[:, 0:4]    # First 4 columns are the features (inputs)
test_classes = test_set[:, 4]     # Last column is the target (classes)

# Displaying the results of the split
print("\nTraining set - inputs:")
print(train_inputs)
print("\nTraining set - classes:")
print(train_classes)
print("\nTest set - inputs:")
print(test_inputs)
print("\nTest set - classes:")
print(test_classes)

# Define the classify_iris function


def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return "Setosa"
    elif pl <= 5:
        return "Virginica"
    else:
        return "Versicolor"


def classify_iris_ADJUSTED(sl, sw, pl, pw):
    if pl < 2 and pw < 1:
        return "Setosa"
    elif sl > 5 and pw > 1.5:
        return "Virginica"
    else:
        return "Versicolor"


# Test the classifier on the test set
good_predictions = 0
len_test = test_set.shape[0]  # Number of records in the test set

for i in range(len_test):
    # Get the features and true class of the current iris
    sl = test_inputs[i, 0]  # sepal length
    sw = test_inputs[i, 1]  # sepal width
    pl = test_inputs[i, 2]  # petal length
    pw = test_inputs[i, 3]  # petal width
    true_class = test_classes[i]  # true species

    # Predict the species using the classify_iris function
    predicted_class = classify_iris_ADJUSTED(sl, sw, pl, pw)

    # Compare the predicted class with the true class
    if predicted_class == true_class:
        good_predictions += 1  # Increment the counter for correct predictions

# Display the results
print("\nNumber of correct predictions:", good_predictions)
print("Percentage of correct predictions:",
      (good_predictions / len_test) * 100, "%")

# Train a decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)

# Evaluate the decision tree classifier
accuracy = dtc.score(test_inputs, test_classes)
print("\nDokładność klasyfikatora drzewa decyzyjnego na zbiorze testowym:", accuracy)

# Display the decision tree in text form
tree_rules = export_text(dtc, feature_names=df.columns[:-1].tolist())
print("\nDrzewo decyzyjne w formie tekstowej:")
print(tree_rules)

# Display the decision tree graphically
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True,
          feature_names=df.columns[:-1].tolist(), class_names=df['variety'].unique().tolist())
plt.savefig("drzewo.png")
plt.show()

# Predict classes for the test set
predictions = dtc.predict(test_inputs)

# Display the confusion matrix
cm = confusion_matrix(test_classes, predictions)
print("\nMacierz błędów:")
print(cm)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel('Przewidywane klasy')
plt.ylabel('Prawdziwe klasy')
plt.title('Macierz błędów')
plt.savefig("maciez.png")
plt.show()

# Compare results
print("\nPorównanie wyników:")
print("Dokładność klasyfikatora drzewa decyzyjnego:", accuracy)
print("Dokładność Twojego ręcznego klasyfikatora:",
      (good_predictions / len_test) * 100, "%")
