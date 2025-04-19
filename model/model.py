import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("data/houses.csv")

# Set the target column
target_column = "House Type"

# Split into features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and encoders
pickle.dump(model, open("model/house_model.pkl", "wb"))
pickle.dump(label_encoders, open("model/label_encoders.pkl", "wb"))
pickle.dump(target_encoder, open("model/target_encoder.pkl", "wb"))

print("✅ Model and encoders saved successfully.")
