from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import math

#reading the given data file
df = pd.read_csv('IOT_Assignment_2_data_regression_sensor_range.csv')

#column renaming to more logical names
df.columns = ['humidity_perc', 'temp_c', 'waterflow_perc']

#min-max normalisation
max_h = df['humidity_perc'].max()
max_t = df['temp_c'].max()

min_h = df['humidity_perc'].min()
min_t = df['temp_c'].min()

df['humidity_perc'] = df['humidity_perc'].apply(lambda x:(x-min_h)/(max_h-min_h) )
df['temp_c'] = df['temp_c'].apply(lambda x: (x-min_t)/(max_t-min_t))


#splitiing the dataframe into inputs and the output
X = df[['humidity_perc', 'temp_c']]
y = df.waterflow_perc

#splitiing the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.25)

# training the MLP model with the following params
# Hidden layer 1: 16 neurons
# Hidden layer 2: 8 neurons
# maximum iterations: 10000
reg = MLPRegressor(hidden_layer_sizes=(16,8),activation="relu" ,random_state=1, max_iter=10000).fit(X_train, y_train)

y_pred=reg.predict(X_test) # prediction using the trained model


#Evaluation metrics
print("The R2 Score with ", (r2_score(y_pred, y_test)))
print("The RMSE with ", (math.sqrt(mean_squared_error(y_pred, y_test))))
print("The mean_absolute_percentage_error  with ", (mean_absolute_percentage_error(y_pred, y_test)))

#final weights and biases
print('The weights are', reg.coefs_)
print('\n\n\n')
print('The biases are', reg.intercepts_)