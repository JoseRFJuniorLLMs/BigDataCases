#path="G:/Freelancing/Machine Learning Videos/Module 4/Data/"

import pandas as pd
from sklearn import preprocessing

df = pd.read_csv(path+"Bike-Sharing-Dataset/day.csv")
df.shape
df.columns
df["season"].unique()
df.head(10)



def label_encoder(df,column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(df[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(temp_array,columns = column_names))


categorical_variables = ["season","yr","mnth","holiday","weekday","workingday","weathersit"]
numeric_variables = ["hum","instant","temp","atemp","windspeed"]
df.head(2)


new_df = df[numeric_variables]
new_df.head(2)
for column in categorical_variables:
    new_df = pd.concat([new_df,label_encoder(df,column)],axis =1)

new_df.shape
new_df.columns
df.columns

from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(new_df,df["cnt"],test_size = 0.3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(X_test)

# RSquare/ Mean Squared Error, Mean Absolute Percentage Error
from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test,y_pred)

model.score(X,y)

mean_squared_error(y_test,y_pred)

sum(abs(y_test -y_pred))/sum(y_test)

w