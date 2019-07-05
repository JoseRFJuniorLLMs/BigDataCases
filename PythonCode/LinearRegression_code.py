import pandas as pd

df = pd.read_csv("G:/Freelancing/Machine Learning Videos/Module 4/Data/Bike-Sharing-Dataset/day.csv")
df.shape
df.columns
df["season"].unique()
df.head(10)

import sklearn.preprocessing
column = "season"
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


X,X_test, Y, Y_test = train_test_split(new_df,df["cnt"],random_state = 2017, test_size = 0.33)
X.index
X.reset_index(inplace=True)
Y = Y.reset_index()

X_test.reset_index(inplace=True)
Y_test = Y_test.reset_index()


Y= Y.cnt.values.reshape(-1,1)
from sklearn import linear_model 
lin_reg = linear_model.LinearRegression()

lin_reg.fit(X,y)
lin_reg.score(X,y)
lin_reg.intercept_
pd.DataFrame(zipX.columns,lin_reg.coef_],columns = ["features","Coeff"])
    
    from sklearn.model_selection import cross_val_predict
    
    predicted = cross_val_predict(lin_reg,X,y,cv=10)

