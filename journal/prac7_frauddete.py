import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Generate synthetic transaction data
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    weights=[0.95, 0.05],
    class_sep=2.0,
    random_state=42
)


# Create DataFrame
data = pd.DataFrame(
    X,
    columns=[
        'amount',
        'time',
        'location_risk',
        'device_risk',
        'num_transactions',
        'balance_change'
    ]
)

data['Class'] = y

print(data.head().round(2))

print("\nClass distribution:")
print(data['Class'].value_counts())


# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

lr = LogisticRegression(
    class_weight='balanced',
    random_state=42
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n========== Logistic Regression ==========")

print("Accuracy:", accuracy_score(y_test, lr_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("Classification Report:")
print(classification_report(y_test, lr_pred))


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n========== Random Forest ==========")

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("Classification Report:")
print(classification_report(y_test, rf_pred))
