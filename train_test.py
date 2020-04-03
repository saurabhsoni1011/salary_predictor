#this is the more acurate one using train_test_split
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

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

lm.fit(x_train,y_train)

y_pred=lm.predict(x_test)

plt.scatter(x_test,y_test)
plt.title("model without prediction and using test data")
plt.xlabel("x test")
plt.ylabel("y test")
sns.regplot(x_test,y_test)
plt.show()

plt.scatter(x_test,y_pred)
plt.title("model without prediction and using test data")
plt.xlabel("x test")
plt.ylabel("y pred")
sns.regplot(x_test,y_pred)
plt.show()

print(lm.score(x_test,y_pred))

ax1=sns.distplot(x_test,y_test,hist=False,color="b",label="actual plot")
sns.distplot(x_test,y_pred,hist=False,color="r",label="predicted plot",ax=ax1)
plt.show()

n=float(input("enter your year of exp to know your salary"))
print(lm.predict([[n]]))