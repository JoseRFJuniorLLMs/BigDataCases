import pandas as pd, numpy as np, matplotlib.pyplot as plt

path = "G:/Freelancing/Machine Learning Videos/Module 5/Data/bank-additional/bank-additional/"

df = pd.read_csv(path+"bank-additional-full.csv",delimiter=";")
df.shape
df.columns
df.dtypes
df.head(10)

df["y"].value_counts()

df.describe()
del(df["duration"])

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def label_encoder(df,column):
    le = preprocessing.LabelEncoder()
    df[column]=le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(df[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(data=temp_array,columns = column_names))


categorical_variables = ["job","poutcome","marital","education","default","housing","loan","contact","month","day_of_week"]
target_variable = ["y"]
numeric_variables = list(set(df.columns.values) - set(categorical_variables) -set(target_variable))


new_df = df[numeric_variables]
for column in categorical_variables:
    new_df= pd.concat([new_df,label_encoder(df,column)],axis=1)
new_df.shape
new_df.columns

target = label_encoder(df,"y")
#Split into test and train 

X, X_test, y , y_test = train_test_split(new_df,target["y_yes"],test_size=0.3,stratify=target["y_yes"])


from xgboost import XGBClassifier

model = XGBClassifier(max_depth = 5, learning_rate = 0.01,n_estimators = 500)
model.fit(X,y)

y_train = model.predict(X)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

accuracy_score(y,y_train)
confusion_matrix(y,y_train)
precision_score(y,y_train)
recall_score(y,y_train)
y.value_counts()

y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
y.value_counts()
model = XGBClassifier()



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, max_depth = 6, min_samples_leaf=15,random_state = 2017,class_weight = {0:0.2,1:0.8})
model.fit(X,y)
model.score(X,y)
y_pred = model.predict(X_test)
confusion_matrix(y_test,y_pred)