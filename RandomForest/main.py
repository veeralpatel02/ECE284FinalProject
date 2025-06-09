import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pipeline import load_dreamt_csv

df = load_dreamt_csv("../data/S002_PSG_df_updated.csv")
df = df.sample(n=5000, random_state=42)
df = df.dropna()
label_col = "Sleep_Stage"
X = df.drop(columns=[label_col])
y = df[label_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))

importances = rf.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
f1s = [report[cls]['f1-score'] for cls in le.classes_]

plt.figure(figsize=(8, 4))
sns.barplot(x=le.classes_, y=f1s)
plt.title("Per-Class F1 Scores")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("../output/f1_per_class_rf.png")
plt.show()

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=le.inverse_transform(y_test), palette="tab10", s=20, alpha=0.7)
plt.title("PCA Projection of Test Set by True Label")
plt.tight_layout()
plt.savefig("../output/pca_scatter.png")
plt.show()
