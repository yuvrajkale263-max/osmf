import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
# Create dataset
data = {
 'file_size': [120, 150, 130, 500, 600, 550,
 110, 140, 520, 580, 125, 610],
 'entropy': [3.1, 3.4, 3.2, 7.2, 7.8, 7.5,
 3.0, 3.5, 7.4, 7.9, 3.3, 8.0],
 'suspicious_imports': [1, 2, 1, 8, 9, 7,
 1, 2, 8, 9, 1, 10],
 'sections': [4, 5, 4, 8, 9, 8,
 4, 5, 9, 8, 4, 10],
 'label': [0, 0, 0, 1, 1, 1,
 0, 0, 1, 1, 0, 1]
}
# Convert dataset into a DataFrame
df = pd.DataFrame(data)
print("Dataset:")
print(df)
# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.25, random_state=42
)
# Create Decision Tree
model = DecisionTreeClassifier(random_state=42)
# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")
# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Test a new file
new_file = pd.DataFrame(
 [[550, 7.6, 8, 9]],
 columns=['file_size', 'entropy', 'suspicious_imports', 'sections']
)
prediction = model.predict(new_file)
if prediction[0] == 1:
 print("New File Prediction: Malware")
else:
 print("New File Prediction: Benign")
