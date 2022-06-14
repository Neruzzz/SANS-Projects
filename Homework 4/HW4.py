import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az

if __name__ == '__main__':
    data = pd.read_csv('datos-17001.csv', sep = ';')
    df = pd.DataFrame({'RefSt': data["RefSt"], 'Sensor_O3': data["Sensor_O3"], 'Temp': data["Temp"], 'RelHum': data["RelHum"]})
    X = df[['Sensor_O3', 'Temp', 'RelHum']]
    Y = df['RefSt']
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, shuffle = True)
    df_train_1 = pd.DataFrame({'Sensor_O3': X_train["Sensor_O3"], 'Temp': X_train["Temp"], 'RelHum': X_train["RelHum"]})
    df_train_y=pd.DataFrame({'RefSt': Y_test})
    df_test_1 = pd.DataFrame({'RefSt': Y_test, 'Sensor_O3': X_test["Sensor_O3"], 'Temp': X_test["Temp"], 'RelHum': X_test["RelHum"]})
    # Loss functions definition
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    def loss_functions(y_true, y_pred):
        print("Loss functions:")
        print("* R-squared =", r2_score(y_true, y_pred))
        print("* RMSE =", mean_squared_error(y_true, y_pred))
        print("* MAE =", mean_absolute_error(y_true, y_pred))
    #plots 1.1
    data.plot(x = 'date', y = 'Sensor_O3', title = 'Sensor_O3 vs date plot')
    plt.ylabel('MOX sensor measurements in [KOhms]')
    plt.show()
    data.plot(x = 'date', y = 'RefSt', title = 'RefSt vs date plot')
    plt.ylabel('Reference station O3 in [ugr/m3]')
    plt.show()
    #plot 1.2
    plt.scatter(data['Sensor_O3'], data['RefSt'])
    plt.title('Sensor_O3 vs RefSt')
    plt.xlabel('MOX sensor measurements in [KOhms]')
    plt.ylabel('Reference station O3 in [ugr/m3]')
    plt.show()
    norSensor_O3 = (data['Sensor_O3']-data['Sensor_O3'].mean())/data['Sensor_O3'].std()
    norRefSt = (data['RefSt']-data['RefSt'].mean())/data['RefSt'].std()
    plt.scatter(norSensor_O3, norRefSt)
    plt.title('Sensor_O3 vs RefSt. NORMALISED')
    plt.xlabel('MOX sensor measurements in [KOhms]')
    plt.ylabel('Reference station O3 in [ugr/m3]')
    plt.show()
    #plot 1.3
    plt.scatter(data['Sensor_O3'], data['Temp'])
    plt.title('Sensor_O3 vs Temp')
    plt.xlabel('MOX sensor measurements in [KOhms]')
    plt.ylabel('Temperature sensor in ºC')
    plt.show()
    plt.scatter(data['Sensor_O3'], data['RelHum'])
    plt.title('Sensor_O3 vs RelHum')
    plt.xlabel('MOX sensor measurements in [KOhms]')
    plt.ylabel('Relative humidity sensor in %')
    plt.show()
    plt.scatter(data['RefSt'], data['Temp'])
    plt.title('RefSt vs Temp')
    plt.xlabel('Reference station O3 in [ugr/m3]')
    plt.ylabel('Temperature sensor in ºC')
    plt.show()
    plt.scatter(data['RefSt'], data['RelHum'])
    plt.title('RefSt vs RelHum')
    plt.xlabel('Reference station O3 in [ugr/m3]')
    plt.ylabel('Relative humidity sensor in %')
    plt.show()
    #MUltiple Linear Regression with normal equations
    #Pred = θ0 + θ1·XSensor_O3 + θ2·XTemp + θ3·XRelHum + variance
    from sklearn.linear_model import LinearRegression
    # Model
    lr = LinearRegression()
    # Fit
    lr.fit(X_train, Y_train)
    # Get MLR coefficients
    #fit_intercept=False sets the y-intercept to 0. If fit_intercept=True, the y-intercept will be determined by the line of best fit.
    print('Intercept: \n', lr.intercept_)
    #coefficent of the linear regression θ1, θ2, θ3
    print('Coefficients: \n', lr.coef_)
    var_RefSt= (data['RefSt'].std()**2)
    # Predict
    df_test_1["MLR_Pred"] = lr.intercept_ + lr.coef_[0]*df_test_1["Sensor_O3"] + lr.coef_[1]*df_test_1["Temp"] + lr.coef_[2]*df_test_1["RelHum"]
    # Plot linear
    plt.scatter(df_test_1["RefSt"],df_test_1["MLR_Pred"])
    plt.xticks(rotation = 20)
    plt.title('RefSt vs MLR_Pred')
    # Plot regression
    sns.lmplot(x='RefSt', y='MLR_Pred', data=df_test_1, fit_reg=True, line_kws={'color': 'orange'})
    #plt.show()
    # Loss
    loss_functions(y_true = df_test_1["RefSt"], y_pred = df_test_1["MLR_Pred"])
    #part 2 Multiple Linear Regression (Beyasian framework)
    basic_model =  pm.Model()
    with basic_model:
        pm.glm.GLM.from_formula("RefSt ~ Sensor_O3 + Temp + RelHum", data, family=pm.families.Normal(), offset=1.0)
        start= pm.find_MAP()
        step=pm.NUTS(scaling=start)
        trace= pm.sample(draws=100, step=step, start=start, progressbar=True)

