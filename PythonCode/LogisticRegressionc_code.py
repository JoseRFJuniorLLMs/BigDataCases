import pandas as pd
new_df = pd.read_csv("G:/ML/Module2/HR_comma_sep.csv")

new_df.shape
new_df.columns

new_df["left"].unique()

# y : left
# x : satisfaction_level

from sklearn.linear_model import LogisticRegression

train =  new_df.iloc[:10000,:]
test  = new_df.iloc[10001:,:]

lor = LogisticRegression()
lor.fit(train["satisfaction_level"].values.reshape(-1,1),train["left"])

lor.predict(test["satisfaction_level"].values.reshape(-1,1))

lor.predict_proba(test["satisfaction_level"].values.reshape(-1,1))

lor.intercept_
lor.coef_


new_df.columns

lor.fit(train[["satisfaction_level","number_project"]],train["left"])

lor.predict(test[["satisfaction_level","number_project"]])

lor.intercept_
lor.coef_


