from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="One-vs-One (OvO) Classifier for Multi-Class Classification with Support Vector Machines (SVM)")
parser.add_argument("test_file", help="Path to the test file in the same format as the training data")
args = parser.parse_args()
penguins = pd.read_csv('penguins_train.csv')
penguins_test = pd.read_csv(args.test_file)

X = penguins.drop('Species', axis=1)
y = penguins['Species']
numeric_column_means = X.select_dtypes(include=['int', 'float']).mean()
X.fillna(numeric_column_means, inplace=True)
cat_columns = X.select_dtypes(include=['object'])
X_encoded = pd.get_dummies(X, columns=cat_columns.columns)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

unique_classes = y.unique()

binary_classifiers = {}

for pair in itertools.combinations(unique_classes, 2):
    class1, class2 = pair
    binary_y_train = y_train[(y_train == class1) | (y_train == class2)]
    binary_X_train = X_train[(y_train == class1) | (y_train == class2)]

    binary_svc = SVC(kernel='linear', probability=True)
    binary_svc.fit(binary_X_train, binary_y_train.map({class1: 0, class2: 1}))

    binary_classifiers[(class1, class2)] = binary_svc

yhat_test = []

for i in range(len(X_test)):
    class_votes = {cls: 0 for cls in unique_classes}

    for class1, class2 in binary_classifiers:
        binary_svc = binary_classifiers[(class1, class2)]
        prob_class1 = binary_svc.predict_proba(X_test.iloc[i:i+1])[0][0]
        prob_class2 = binary_svc.predict_proba(X_test.iloc[i:i+1])[0][1]

        if prob_class1 > prob_class2:
            class_votes[class1] += 1
        else:
            class_votes[class2] += 1

    predicted_class = max(class_votes, key=class_votes.get)
    yhat_test.append(predicted_class)

accuracy = accuracy_score(y_test, yhat_test)
predictions_df = pd.DataFrame(data={'predicted': yhat_test})
predictions_df.to_csv('ovo.csv', index=False)
print(f"Predictions saved to 'ovo.csv'.")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, yhat_test)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and display the classification report (precision, recall, F1-score)
class_names = unique_classes
report = classification_report(y_test, yhat_test, target_names=class_names)
print("Classification Report:")
print(report)