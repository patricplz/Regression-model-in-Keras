#imports

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# load the data into the code
df = pd.read_csv('concrete_data.csv')


# Get all column names
df_columns = df.columns

# Separate features and target variable
features = df[df_columns[df_columns != 'Strength']]  # All columns except 'Strength'
target = df['Strength']  # Target variable

# function to create the neural network model
def create_neural_network():
    model = Sequential()
    model.add(Dense(10, input_dim=features.shape[1], activation='relu'))  # Hidden layer with 10 nodes
    model.add(Dense(1))  # Output layer with 1 single node
    model.compile(optimizer=Adam(), loss='mean_squared_error')  # Configure optimizer (adam) and loss function (mse)
    return model

# number of repetitions
num_repeats = 50
mse_results = []

# repeat the process 50 times as we said before
for _ in range(num_repeats):
    # Split the data into training and testing sets with 30% of test data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=np.random.randint(0, 10000))
    
    # create and train the model
    model = create_neural_network()
    model.fit(X_train, y_train, epochs=50, verbose=0)  # Train with 50 epochs
    
    # make predictions and calculate MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mse_results.append(mse)

# calculate and display results
avg_mse = np.mean(mse_results)
print(f"Average: {avg_mse}")

mse_variance = np.std(mse_results)
print(f"Standard Deviation: {mse_variance}")


#my solution:
#Average: 362.38086207497486
#Standard Deviation: 461.5911607826393