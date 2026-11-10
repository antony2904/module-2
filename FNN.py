import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Training data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[0], [0], [0], [1], [1], [1]])

# Create FNN model
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=100, verbose=0)

# Predict
prediction = model.predict(np.array([[5]]), verbose=0)

print("Prediction:", prediction[0][0])
print("Result:", "Pass" if prediction[0][0] > 0.5 else "Fail")