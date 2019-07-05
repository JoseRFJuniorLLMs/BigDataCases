
import pandas as pd

df = pd.read_csv("G:/ML/Module2/HR_comma_sep.csv")
type(df)

df.shape
df.columns

df.head(5)
df.tail(5)

df.info()

df.describe()

df["satisfaction_level"]
df2 = df[["satisfaction_level","salary","sales"]]

df_copy = df.iloc[:500,[0,3,9]]

df["col1"]=1
df["impact"]=df["satisfaction_level"]* df["number_project"]

del df["col1"]

df["new_salary"]= df["salary"].str.upper()+ " - New"

df.describe()
df_new = df[ (df["number_project"] > 4) & (df["salary"]=="low")]

df_new["number_project"].min()

df["salary"].unique()

df_new1=df.sort_values(["impact","number_project"])

#Merge, join, concat and pivot, groupby


#merge
left =df
right = df.groupby(["sales"],as_index=False)["satisfaction_level"].mean()
right.columns =["sales","satifaction_index"]

df_merged = pd.merge(left,right,how="inner",on=["sales"])

#Join
left1 = df.iloc[:1500,:4]
right1 = df.iloc[:500,7:]

joined = left1.join(right1,how = "left")

#Concat
df_grouped1 = df.groupby(["sales"],as_index=False)["satisfaction_level"].mean()

df_grouped2 = df.groupby(["sales"],as_index=False)["satisfaction_level"].sum()

concated= pd.concat([df_grouped1,df_grouped2],axis=1)
concated.shape

#Pivot
df_pvt = df.groupby(["sales","salary"],as_index=False)["satisfaction_level"].mean()
df_pvt2 = df_pvt.pivot(index="sales",columns ="salary",values="satisfaction_level")
