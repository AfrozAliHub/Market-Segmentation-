import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your data
data = pd.read_csv('Clustered_Customer_Data.csv')  
X = data.drop('Cluster', axis=1)  
y = data['Cluster'] 

# Ensure column names are uppercase if required
X.columns = [col.upper() for col in X.columns]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()  
model.fit(X_train, y_train)

# Save the model to a file
with open('final_model.pkl', 'wb') as f:
    pickle.dump(model, f)
