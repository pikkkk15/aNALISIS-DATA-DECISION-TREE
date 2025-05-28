import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load Dataset ===
data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(data_path)

# === Preprocessing ===
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# === Split Data ===
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)

# === Evaluation ===
y_pred = clf.predict(X_test)

# Print the Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

# Print Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Additional Evaluation Metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# === Visualisasi Pohon ===
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(i) for i in sorted(y.unique())], filled=True)
plt.savefig("decision_tree_pegawai.png")
plt.show()

# === Confusion Matrix Visualization ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in sorted(y.unique())], yticklabels=[str(i) for i in sorted(y.unique())])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix_pegawai.png")
plt.show()
