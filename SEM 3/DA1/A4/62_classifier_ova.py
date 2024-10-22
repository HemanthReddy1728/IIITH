from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

parser = argparse.ArgumentParser(description="One-vs-All (OvA) Classifier for Multi-Class Classification with Support Vector Machines (SVM)")
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

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=31)

unique_classes = y.unique()

ova_classifiers = {}

for target_class in unique_classes:
    binary_y_train = (y_train == target_class).astype(int) 
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X_train, binary_y_train)
    
    ova_classifiers[target_class] = svc

yhat_test = []

for i in range(len(X_test)):
    class_scores = {cls: binary_svc.predict_proba(X_test.iloc[i:i+1])[0][1] for cls, binary_svc in ova_classifiers.items()}
    
    predicted_class = max(class_scores, key=class_scores.get)
    yhat_test.append(predicted_class)

accuracy = accuracy_score(y_test, yhat_test)
predictions_df = pd.DataFrame(data={'predicted': yhat_test})
predictions_df.to_csv('ova.csv', index=False)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, yhat_test)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and display the classification report (precision, recall, F1-score)
report = classification_report(y_test, yhat_test, target_names=unique_classes)
print("Classification Report:")
print(report)
