import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pymc3 as pm

############################## FUNCTIONS ####################################

def metrics(y_tr, y_pr): # Prints the metrics of the model
    print('Metrics:')
    print('- R^2 = ', r2_score(y_tr, y_pr))
    print('- RMSE = ',np.sqrt(mean_squared_error(y_tr, y_pr)))
    print('- MAE = ', mean_absolute_error(y_tr, y_pr))

def norm(column): # Normalizes the colum, returns the value of the normalization
    mu = column.mean()
    sigma = column.std()
    return (column - mu)/sigma

############################### CODE ##########################################

# Create data with all the CSV
data = pd.read_csv('datos-17001.csv', sep = ';', index_col = 0, parse_dates = False)

# Create the dataframe
df = pd.DataFrame({'RefSt': data["RefSt"], 'Sensor_O3': data["Sensor_O3"], 'Temp': data["Temp"], 'RelHum': data["RelHum"]})

# Splitting the dataset into Y: RefSt and X: Sensor_O3, Temp and RelHum
X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

# Shuffling the data and assigning 30% to the test set and 70% to the train set
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size = 0.3, random_state = 1, shuffle = True)


########## 2.1. Plot Sensor_O3 and RefSt as function of time
df[["RefSt", "Sensor_O3"]].plot()
plt.xticks(rotation = 20)

########## 2.2. Scatter plot of Sensor_O3 and RefSt, and the same with normalization
sp.lmplot(x = 'Sensor_O3', y = 'RefSt', data = df, fit_reg = False) 

df["normRefSt"] = norm(df["RefSt"])
df["normSensor_O3"] = norm(df["Sensor_O3"])

sp.lmplot(x = 'normSensor_O3', y = 'normRefSt', data = df, fit_reg = False)


########## 2.3. Scatterplots Sensor/RefSt with respect to the Temperature and the Humidity (Normalised)

df["normTemp"] = norm(df["Temp"])
df["normRelHum"] = norm(df["RelHum"])

sp.lmplot(x = 'normSensor_O3', y = 'normTemp', data = df, fit_reg = False)

sp.lmplot(x = 'normSensor_O3', y = 'normRelHum', data = df, fit_reg = False)

sp.lmplot(x = 'normRefSt', y = 'normTemp', data = df, fit_reg = False)

sp.lmplot(x = 'normRefSt', y = 'normRelHum', data = df, fit_reg = False)



########## 3. Calibration using multiple linear regression (frequentist framework)

df_tr = pd.DataFrame({'RefSt': Y_tr, 'Sensor_O3': X_tr["Sensor_O3"], 'Temp': X_tr["Temp"], 'RelHum': X_tr["RelHum"]})
df_te = pd.DataFrame({'RefSt': Y_te, 'Sensor_O3': X_te["Sensor_O3"], 'Temp': X_te["Temp"], 'RelHum': X_te["RelHum"]})

lr = LinearRegression()
lr.fit(X_tr, Y_tr)

# Printing the coefficients
print('Intercept = ',lr.intercept_)
print('Coefficient 1 = ',lr.coef_[0])
print('Coefficient 2 = ',lr.coef_[1])
print('Coefficient 3 = ',lr.coef_[2])

# Multiple Linear Regression formula Pred = θ0+ θ1·XSensor_O3 + θ2·XTemp + θ3·XRelHum + variance
df_te["MLR_Pred"] = lr.intercept_ + lr.coef_[0]*df_te["Sensor_O3"] + lr.coef_[1]*df_te["Temp"] + lr.coef_[2]*df_te["RelHum"]

# Scatter plot and regression line for the test set
sp.lmplot(x='RefSt', y='MLR_Pred', data=df_te, fit_reg=True, line_kws={'color': 'red'})
plt.title('RefSt vs MLR_Pred (TEST SET)')


# Printing the metrics for test set
print('Metrics for the test set')
metrics(y_tr = df_te["RefSt"], y_pr = df_te["MLR_Pred"])

# Printing the metrics for training set
df_tr["MLR_Pred"] = lr.intercept_ + lr.coef_[0]*df_tr["Sensor_O3"] + lr.coef_[1]*df_tr["Temp"] + lr.coef_[2]*df_tr["RelHum"]
print('Metrics for the training set')
metrics(y_tr = df_tr["RefSt"], y_pr = df_tr["MLR_Pred"])

# Scatter plot and regression line for the training set
sp.lmplot(x='RefSt', y='MLR_Pred', data=df_tr, fit_reg=True, line_kws={'color': 'red'})
plt.title('RefSt vs MLR_Pred (TRAINING SET)')


# 
df_te[["RefSt", "MLR_Pred"]].plot()
plt.xticks(rotation = 20)

plt.show()


########## 3. Calibration using multiple linear regression (frequentist framework)
basic_model =  pm.Model()
with basic_model:
    pm.glm.GLM.from_formula("RefSt ~ Sensor_O3 + Temp + RelHum", data, family=pm.families.Normal(), offset=1.0)
    start= pm.find_MAP()
    step=pm.NUTS(scaling=start)
    trace= pm.sample(draws=100, step=step, start=start, progressbar=True)