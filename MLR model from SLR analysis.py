# Import all libraries---------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#------------------------------


# Finding Correlation Coefficient------------------------
class coefficient:              
                       
    def corrcoeff(X, Y):
        
        m = list()                      
        n = list()     
                                                
        for i in np.array(X):           
            for j in i:                 
                m.append(j)             
                                        
        for i in np.array(Y):           
            n.append(i)                 
                                        
        r = np.corrcoef(m, n)   
        
        # r[0, 1] is correlation coefficient        
        # print(r[0, 1])                  
        # r[0, 1]*r[0, 1] is coefficient of determination
        
        K = [m, n, r[0, 1]*r[0, 1]]
        return K
# -------------------------------------------------------



# Regression line slope and intercept function ----------------------------------------------------
def slope_intercept(x, y):
    x1 = np.array(x)
    y1 = np.array(y)
    
    m = ((np.mean(x1)*np.mean(y1)) - np.mean(x1*y1)) / ((np.mean(x1)*np.mean(x1)) - np.mean(x1*x1))
        
    m = round(m, 2)
    b = (np.mean(y1) - np.mean(x1)*m)
    b = round(b, 2)
    
    return m, b
# -------------------------------------------------------------------------------------------------
    


# Import the dataset and set dependent and independent variables
dataset = pd.read_csv('50_Startups.csv')

# Independent variable
X1 = dataset.iloc[:, 0:1]
X2 = dataset.iloc[:, 1:2]
X3 = dataset.iloc[:, 2:3]

# Dependent variable
Y = dataset.iloc[:, -1]
# --------------------------------------------------------------



# Construct regressor ---------------------------
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X1, Y)
Y1_pred = regressor.predict(X1)


regressor.fit(X2, Y)
Y2_pred = regressor.predict(X2)


regressor.fit(X3, Y)
Y3_pred = regressor.predict(X3)
# -----------------------------------------------


# Visualise data ---------------------------------
plt.scatter(X1, Y, color = 'red', label = 'R&D v/s Profit')
plt.plot(X1, Y1_pred, color = 'blue', label = 'SLR line')
plt.legend(loc = 'upper left')
plt.xlabel('R&D Spend')
plt.ylabel('Profit value')
plt.show()


plt.scatter(X2, Y, color = 'red', label = 'Admin exp v/s Profit')
plt.plot(X2, Y2_pred, color = 'blue', label = 'SLR line')
plt.legend(loc = 'upper left')
plt.xlabel('Administration expenditure')
plt.ylabel('Profit value')
plt.show()


plt.scatter(X3, Y, color = 'red', label = 'Market exp v/s Profit')
plt.plot(X3, Y3_pred, color = 'blue', label = 'SLR line')
plt.legend(loc = 'upper left')
plt.xlabel('Marketing Expenditure')
plt.ylabel('Profit value')
plt.show()
# ------------------------------------------------



# Calculate coefficients and return 1d array of IDV and DV
cc_X1_Y = coefficient.corrcoeff(X1, Y)
cc_X2_Y = coefficient.corrcoeff(X2, Y)
cc_X3_Y = coefficient.corrcoeff(X3, Y)
# --------------------------------------------------------



# Print correlation Coefficients---------------------------------------------------
print(cc_X1_Y[2])          # Proves R&D expediture increases profit at a high rate
print(cc_X2_Y[2])          # Adminstration expenditure has little to do with profit
print(cc_X3_Y[2])          # Markettig expenditure is partially profitable
# ---------------------------------------------------------------------------------


print(cc_X1_Y[0][0])

# Finding the equation of the regression line -----------------------------------
print("Equation of R&D v/s profit regression line")
m1, b1 = slope_intercept(cc_X1_Y[0], cc_X1_Y[1])
print("slope = ", m1,", Intercept = ", b1)
print("y = ", m1, "*x1", "+", b1)

print("\n---------------------------------------------------------------------\n")

print("Equation of Administration expenditure v/s profit regression line")
m2, b2 = slope_intercept(cc_X2_Y[0], cc_X2_Y[1])
print("slope = ", m2,", Intercept = ", b2)
print("y = ", m2, "*x2", "+", b2)

print("\n---------------------------------------------------------------------\n")

print("Equation of Marketing expense v/s Profit regression line")
m3, b3 = slope_intercept(cc_X3_Y[0], cc_X3_Y[1])
print("slope = ", m3,", Intercept = ", b3)
print("y = ", m3, "*x3", "+", b3)
# --------------------------------------------------------------------------------



# Alternative MLR Algorithm to determine profit -----------------------------------------------

""" Adding the 3 equations gives 3y = 0.85*x1 + 0.29*x2 + 0.25*x3 + 185428.32.
    However this is considering equal weights to x1, x2 and x3, which is not true
    their correlation coefficients are not equal to 1. This means x1, x2, x3 have
    a weighted contribution to y. Let r1, r2, r3 be the weights.
    
    Then the equation will be:
        
        r1*y + r2*y + r3*y = 0.85*r1*x1 + 0.29*r2*x2 + 0.25*r2*x3 + 
                             49349.27*r1 + 76822.69*r2 + 59256.36*r3
        
    r1, r2, r3 are the correlation coefficients of x1 & y, x2 & y and x3 & y 
    respectively."""
    
def profit_MLR(l, m, n):
    
    profit_pred = []
    
    for i in range(0, 50):
        
        x1 = l[i]
        x2 = m[i]
        x3 = n[i]
        
        Y = ((0.85*cc_X1_Y[2]*x1 + 0.29*cc_X2_Y[2]*x2 + 0.25*cc_X3_Y[2]*x3 + 82937.418)/1.5459)
    
        profit_pred.append(Y)
    
    return profit_pred  
    
MLR_predict = profit_MLR(cc_X1_Y[0], cc_X2_Y[0], cc_X3_Y[0])

x = [x for x in range(1, 51)]

plt.scatter(x, Y, color = 'red', label = 'Actual Profit')
plt.plot(x, MLR_predict, color = 'blue', label = 'Predicted Profit')
plt.legend(loc = "upper right")
plt.title("MLR from SLR analysis")
plt.xlabel("Index")
plt.ylabel("Profit value")
plt.show()

# ---------------------------------------------------------------------------------------------

# Show correlation accuracy -------------------------------------------------------
print("From Actual profit and predicted profit we can correlation between them = ", 
      np.corrcoef(Y, MLR_predict)[0, 1]*100, "%")
# ---------------------------------------------------------------------------------


