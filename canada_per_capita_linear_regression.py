import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("canada_per_capita_income.csv")
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='*')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
my_prediction = reg.predict([[2022]])
print(my_prediction)
print(reg.coef_)
print(reg.intercept_)
plt.plot(df['year'], reg.predict(df[['year']]), color='blue')
plt.show()
