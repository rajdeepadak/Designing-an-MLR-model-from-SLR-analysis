# Importing Libraries ---------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# -----------------------------


# Construct dataset and IDV, DV --------
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
# --------------------------------------


# Categorical Encoding ------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
# ---------------------------------------------------------------------------------------------------


# Perform Multiple Linear Regression ------------
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)
y_pred = regressor.predict(X)
# -----------------------------------------------


# Visualize data -------------------------------------------------------------
Y = np.array(Y)
x = [x for x in range(1, 51)]
plt.scatter(x, Y, color = 'red', label = 'Actual Profit')
plt.plot(x, y_pred, color = 'blue', label = 'Predicted Profit (standard MLR)')
plt.title("Direct MLR analysis")
plt.legend(loc = 'upper right')
plt.xlabel('Index')
plt.ylabel('Profit value')
plt.show()
# ----------------------------------------------------------------------------

# Show correlation accuracy ----------------
print(np.corrcoef(Y, y_pred)[0, 1]*100, "%")
# ------------------------------------------