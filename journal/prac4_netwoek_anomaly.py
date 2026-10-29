import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


np.random.seed(42)


# ------------------------------------------
# Step 1: Generate 500 normal records
# ------------------------------------------

normal = pd.DataFrame({
    'duration': np.random.normal(5, 1.5, 500),
    'bytes_sent': np.random.normal(500, 100, 500),
    'connections': np.random.normal(10, 3, 500),
    'label': 0
})


# ------------------------------------------
# Step 2: Generate 25 attack records
# ------------------------------------------

attack = pd.DataFrame({
    'duration': np.random.normal(30, 5, 25),
    'bytes_sent': np.random.normal(5000, 800, 25),
    'connections': np.random.normal(120, 20, 25),
    'label': 1
})


# ------------------------------------------
# Step 3: Combine the datasets
# ------------------------------------------

data = pd.concat(
    [normal, attack],
    ignore_index=True
)

print(data.head())

print("\nClass distribution:")
print(data['label'].value_counts())


# ------------------------------------------
# Step 4: Visualize actual records
# ------------------------------------------

colors = data['label'].map({
    0: 'blue',
    1: 'red'
})

plt.scatter(
    data['bytes_sent'],
    data['connections'],
    c=colors,
    alpha=0.6
)

plt.xlabel('Bytes Sent')
plt.ylabel('Connections')
plt.title('Blue = Normal, Red = Attack')
plt.show()


# ------------------------------------------
# Step 5: Prepare training and testing data
# ------------------------------------------

X = data[
    ['duration', 'bytes_sent', 'connections']
]

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ------------------------------------------
# Step 6: Decision Tree classifier
# ------------------------------------------

clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, pred))


# ------------------------------------------
# Step 7: Isolation Forest anomaly detection
# ------------------------------------------

iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Train without using the actual labels
iso.fit(X)

# Isolation Forest: 1 = normal, -1 = anomaly
data['pred'] = iso.predict(X)

# Convert to 0 = normal, 1 = anomaly
data['pred'] = data['pred'].map({
    1: 0,
    -1: 1
})


print("\nIsolation Forest Confusion Matrix:")
print(confusion_matrix(data['label'], data['pred']))

print("\nIsolation Forest Classification Report:")
print(classification_report(data['label'], data['pred']))


# ------------------------------------------
# Step 8: Visualize detected anomalies
# ------------------------------------------

colors = data['pred'].map({
    0: 'blue',
    1: 'red'
})

plt.scatter(
    data['bytes_sent'],
    data['connections'],
    c=colors,
    alpha=0.6
)

plt.xlabel('Bytes Sent')
plt.ylabel('Connections')
plt.title('Isolation Forest: Red = Detected Anomaly')
plt.show()