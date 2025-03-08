import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/movies.csv')
print(data.head()) # [1000 rows x 2 columns]

plt.figure(figsize=(10, 6))
plt.scatter(data.production_budget_usd, data.worldwide_gross_usd, alpha=0.4)
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.xlim(0, 299824000)
plt.ylim(0, 2076734000)

model = LinearRegression()
X = pd.DataFrame(data.production_budget_usd)
y = pd.DataFrame(data.worldwide_gross_usd)
model.fit(X, y)

print(f"Coef = {model.coef_}") # Coef = [[4.38435429]]
print(f"Intercept = {model.intercept_}") # Intercept = [-8431387.34127641]
print(f"Score = {model.score(X, y)}") # Score = 0.6449510860262893

budget = np.array([[100000000]])
predict = model.predict(budget) # [[4.30004042e+08]]
result = (((predict/budget) - 1).round()*100).item()
print(f"Net income = {int(result)}%") # Net income = 300%


if __name__ == "__main__":
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    plt.show()
