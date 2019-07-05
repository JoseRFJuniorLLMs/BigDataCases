#regression - HR Analytics 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
new_df = pd.read_csv("G:/ML/Module2/HR_comma_sep.csv")


new_df.shape
new_df.columns

new_df["Work_accident"].unique()
new_df["Work_accident"] = new_df["Work_accident"].astype("category")

new_df.dtypes

new_df["salary"].unique()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
new_df["salary2"] = le.fit_transform(new_df["salary"])

new_df["salary2"] = new_df["salary2"].astype("category") 

dt = DecisionTreeClassifier()

dt.fit(new_df[["salary2","Work_accident"]],new_df["left"])

dt.predict(new_df[["salary2","Work_accident"]])

from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder()
temp = ohe.fit_transform(new_df[["salary2"]]).toarray()

column_names = ["salary_"+x for x in le.classes_]

temp = pd.DataFrame(temp,columns = column_names)

temp.head(10)

new_df2 = pd.concat([new_df,temp],axis=1)
new_df2.head(10)

new_df2.shape

export_graphviz(dt,"G:/ML/Module2/test.dot",feature_names = ["salary2","Work_accident"])
