import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("G:/ML/Module2/HR_comma_sep.csv")

df.describe()


df["satisfaction_level"].describe()

df["satisfaction_level"].plot(kind="box")


plt.boxplot(df["satisfaction_level"],showmeans=True)


df["satisfaction_level"].plot(kind="hist")


plt.hist(df["satisfaction_level"],bins=30)


df["satisfaction_level"].var()
df["satisfaction_level"].std()
df["satisfaction_level"].mean()

df["satisfaction_level"].std()/df["satisfaction_level"].mean() *100

df["satisfaction_level"].skew()

df["satisfaction_level"].kurtosis()




df["sales"].unique()

df["salary"].unique()
pd.crosstab(df["sales"],df["salary"]).plot(kind="bar",stacked = True)

df.columns

#Numeric and Categocrical variable exploration
df.groupby(["salary"])["satisfaction_level"].mean().plot(kind="bar")

df.corr()

plt.scatter(df["average_montly_hours"],df["time_spend_company"])


df.shape




df.groupby(["salary"])["satisfaction_level"].mean()

"T Test"
from scipy.stats import ttest_ind
col1 = df[df["salary"]=="high"]
col2 = df[df["salary"]=="low"]

ttest_ind(col1["satisfaction_level"],col2["satisfaction_level"])

