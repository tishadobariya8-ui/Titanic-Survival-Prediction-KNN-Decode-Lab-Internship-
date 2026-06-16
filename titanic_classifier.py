import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline

# Load Dataset
df = pd.read_csv("data/Titanic_dataset.csv")

# Remove unnecessary columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

print(df.shape)

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Convert categorical data to numeric
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Display dataset information
print("First 5 Records:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Shape:", df.shape)

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nFeatures Used:", list(X.columns))

# Survival Distribution plot
df["Survived"].value_counts().plot(
    kind="bar",
    title="Survival Distribution"
)
plt.xlabel("Survived")
plt.ylabel("Count")
plt.savefig("survival_distribution.png")
plt.close()
print(df["Survived"].value_counts())

# Train-Test Split (with stratify)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)
print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Testing Shape:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)
print("\nScaling Completed!")

# Find Best K Value
best_k = 1
best_score = 0
k_values = []
accuracies = []

for k in range(1, 21):
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train_scaled, y_train)
    score = temp_model.score(X_val_scaled, y_val)
    k_values.append(k)
    accuracies.append(score)
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nOptimal K Selected: {best_k}")
print(f"Best Accuracy During Search: {best_score:.4f}")

# K vs Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker="o")
plt.title("K Value vs Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("k_vs_accuracy.png")
plt.close()
print("Plot saved: k_vs_accuracy.png")

# Train Model
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_full_scaled, y_train_full)
print(f"\nFinal model trained with K={best_k}")

# Prediction
y_pred = model.predict(X_test_scaled)
print("\nSample Predictions:", y_pred[:10])

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
train_accuracy = model.score(X_train_full_scaled, y_train_full)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Results
print("\nTraining Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy:", round(accuracy, 4))

# Fixed: Confusion Matrix printed clearly
print("\nConfusion Matrix:")
print(f"TN={cm[0][0]} FP={cm[0][1]}")
print(f"FN={cm[1][0]} TP={cm[1][1]}")

print("\nF1 Score:", round(f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross Validation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:", cv_scores)

print("\nAverage Cross Validation Accuracy:", round(cv_scores.mean(), 4))

print("\nProject Completed Successfully!")
print(f"Final Accuracy: {accuracy * 100:.2f}%")
