import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv('loan_prediction.csv')
data = data.drop('Loan_ID',axis=1)
columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.dropna(subset=columns)
data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')
data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
# Replace missing values in 'Self_Employed' column with a default value (e.g., 'Unknown')
data['Self_Employed'].fillna('Unknown', inplace=True)

# Map 'Yes' and 'No' values to 1 and 0, respectively
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0, 'Unknown': 0}).astype(int)

data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

from sklearn.preprocessing import StandardScaler
sw = StandardScaler()
X[cols]=sw.fit_transform(X[cols])
# Load the dataset
model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)

from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
RandomForestClassifier()
rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)
rs_rf.fit(X,y)
# Create a Streamlit app
st.title('Loan Status Prediction')

# Sidebar for user input
st.sidebar.header('Input Features')
applicant_income = st.sidebar.number_input('Applicant Income', min_value=0)
coapplicant_income = st.sidebar.number_input('Coapplicant Income', min_value=0)
loan_amount = st.sidebar.number_input('Loan Amount', min_value=0)
loan_term = st.sidebar.number_input('Loan Term', min_value=0)
credit_history = st.sidebar.selectbox('Credit History', [0, 1])

# Make a prediction
prediction = rs_rf.predict(X)

# Display the prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Congratulations! Your loan is likely to be approved.')
else:
    st.write('Sorry, your loan is likely to be rejected.')

# Display the dataset
if st.checkbox('Show Dataset'):
    st.subheader('Loan Prediction Dataset')
    st.write(data)
