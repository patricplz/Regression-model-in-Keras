import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the data into the code
df = pd.read_csv('concrete_data.csv')


# Separate features and target variable
features = df.drop(columns='Strength')  # Features
target = df['Strength']  # Target variable

# Function to create the neural network model
def create_neural_network():
    model = Sequential()
    model.add(Dense(10, input_dim=features.shape[1], activation='relu'))  # Hidden layer with 10 nodes
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(), loss='mean_squared_error')  # Configure optimizer and loss function
    return model

# Number of repetitions
num_repeats = 50
mse_results = []

# Repeat the process 50 times
for _ in range(num_repeats):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=np.random.randint(0, 10000))
    
    # Create and train the model
    model = create_neural_network()
    model.fit(X_train, y_train, epochs=50, verbose=0)  # Train with 50 epochs
    
    # Make predictions and calculate MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mse_results.append(mse)

# Calculate and display results
avg_mse = np.mean(mse_results)
print(f"Average: {avg_mse}")

mse_variance = np.std(mse_results)
print(f"Standard Deviation: {mse_variance}")
