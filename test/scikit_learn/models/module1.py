import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset from txt
data = np.loadtxt(datasets, skiprows=1)  # skip header
X = data[:, :3]  # first 3 columns = features
y = data[:, 3]   # last column = label

# 2. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale (very important for neural nets)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Small Neural Network Model
# hidden_layer_sizes=(10,) = 1 hidden layer with 10 neurons
# hidden_layer_sizes=(8, 5) = 2 hidden layers (8 neurons, then 5)
model = MLPClassifier(
    hidden_layer_sizes=(10,),  # SUPER SMALL MODEL
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.1f}%")
print(classification_report(y_test, pred))

# 7. Predict new student: [study=6h, sleep=7h, prev=70]
new_data = scaler.transform([[6, 7, 70]])
print("Prediction for new student:", model.predict(new_data))
