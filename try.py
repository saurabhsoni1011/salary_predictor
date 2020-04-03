import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv("C:\\Users\\DELL\\Desktop\\project_1_dataset.csv")
df.head()
lm=LinearRegression()
x=df[["YearsExperience"]]
y=df["Salary"]
plt.scatter(x,y)
plt.title("scatter plot")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
sns.regplot(x,y,data=df)
plt.show()

lm.fit(x,y)
print(lm.score(x,y)) #r-sq value
y_pred=lm.predict(x)
plt.scatter(x,y_pred)

plt.xlabel("Years of Experience")
plt.ylabel("predicted Salary")
plt.show()

ax1=sns.distplot(y,hist=False,color="r",label="actual value")
sns.distplot(y_pred,hist=False,color="b",label="predicted value",ax=ax1)
plt.show()

n=float(input("enter your years of exp"))
print(lm.predict([[n]]))