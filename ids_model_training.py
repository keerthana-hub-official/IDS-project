import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the dataset
columns = [f'feature{i}' for i in range(1, 42)] + ['label']
train_data = pd.read_csv("KDDTrain+.txt", names=columns)
test_data = pd.read_csv("KDDTest+.txt", names=columns)

# Combine both for training and testing
data = pd.concat([train_data, test_data])

# Step 2: Convert labels to binary: 'normal' or 'attack'
data['label'] = data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Step 3: Encode all values to numeric
encoder = LabelEncoder()
for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

# Step 4: Split data
X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Test the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the trained model
joblib.dump(model, "ids_model.pkl")
print("✅ Model saved as 'ids_model.pkl'")
