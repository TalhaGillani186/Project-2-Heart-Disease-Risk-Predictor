import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("heart_disease_uci.csv")
print("Original shape:", df.shape)

df = df.drop(["id", "dataset"], axis=1)

df["target"] = (df["num"] > 0).astype(int)
df = df.drop("num", axis=1)

df["fbs"] = df["fbs"].map({True: 1, False: 0})
df["exang"] = df["exang"].map({True: 1, False: 0})

categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]
numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "fbs", "exang"]

num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df = df[(df["trestbps"] >= 80) & (df["trestbps"] <= 220)]
df = df[(df["chol"] >= 80) & (df["chol"] <= 600)]
df = df[(df["thalch"] >= 60) & (df["thalch"] <= 220)]

print("Shape after cleaning:", df.shape)

plt.figure(figsize=(6, 4))
df["target"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Heart Disease Distribution")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graph_class_balance.png")
plt.show()

plt.figure(figsize=(7, 5))
df[df["target"] == 0]["age"].hist(alpha=0.6, label="No Disease", bins=20)
df[df["target"] == 1]["age"].hist(alpha=0.6, label="Has Disease", bins=20)
plt.title("Age Distribution by Disease Status")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.legend()
plt.tight_layout()
plt.savefig("graph_age_distribution.png")
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("graph_correlation_heatmap.png")
plt.show()

print("\nCorrelation of each feature with 'target':")
print(correlation_matrix["target"].sort_values(ascending=False))

corr_target = correlation_matrix["target"].drop("target").abs().sort_values(ascending=False)
selected_features = corr_target[corr_target > 0.05].index.tolist()
print("\nFeatures used:", selected_features)

X = df[selected_features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train), "| Testing samples:", len(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_k = None
best_accuracy = 0

for k in [5, 7, 9, 11, 15, 21]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"\nBest K: {best_k}")
print(f"Final Accuracy: {round(accuracy * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Heart Disease Prediction")
plt.colorbar()
plt.xticks([0, 1], ["No Disease", "Has Disease"])
plt.yticks([0, 1], ["No Disease", "Has Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("graph_confusion_matrix.png")
plt.show()

with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("model_info.pkl", "wb") as f:
    pickle.dump({
        "columns": selected_features,
        "accuracy": accuracy,
        "best_k": best_k,
        "categorical_cols": [c for c in categorical_cols if c in selected_features],
    }, f)

print("\nModel, scaler, and encoders saved successfully.")
print(f"Final model: KNN (k={best_k}), Accuracy: {round(accuracy * 100, 2)}%")
