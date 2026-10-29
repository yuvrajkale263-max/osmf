import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Generate sample data
np.random.seed(42)

genuine = pd.DataFrame({
    'typing_speed': np.random.normal(60, 5, 200),
    'key_hold_time': np.random.normal(0.15, 0.02, 200),
    'key_gap_time': np.random.normal(0.30, 0.04, 200),
    'label': 1
})

imposter = pd.DataFrame({
    'typing_speed': np.random.normal(45, 8, 200),
    'key_hold_time': np.random.normal(0.22, 0.04, 200),
    'key_gap_time': np.random.normal(0.45, 0.08, 200),
    'label': 0
})

# Combine the data
data = pd.concat([genuine, imposter]).reset_index(drop=True)

print(data.head())


# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)


# Make predictions
y_pred = knn.predict(X_test_scaled)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))


# Test a new typing attempt
new_attempt = pd.DataFrame(
    [[58, 0.16, 0.31]],
    columns=X.columns
)

# Scale the new attempt using the SAME scaler
new_attempt_scaled = scaler.transform(new_attempt)

# Predict
result = knn.predict(new_attempt_scaled)

print("Access Granted" if result[0] == 1 else "Access Denied")
