######################################## importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

######################################### importing dataset
car=pd.read_csv("C:/Users/yamini/Desktop/GitHub/Naive Bayes/Bernoulli/NB_Car_Ad.csv")
car

car.isnull().sum()                  #no null values
car.duplicated().sum()              #no duplicate values

car.columns
car1=car.drop(["User ID"],axis=1)
car1

########################################## One Hot Encoding for categorical variables
car1.columns
enc=OneHotEncoder(handle_unknown="ignore")
enc_var=pd.DataFrame(enc.fit_transform(car1[['Gender']]).toarray())
enc_var.head(5)
enc.get_feature_names_out(['Gender'])

#changing names for discrete variables
enc_var=enc_var.rename(columns={0:'Gender_Female', 1:'Gender_Male'})
enc_var

########################################## joining both continuous and discrete data
car2=car1.join(enc_var)
car2.head(2)

car2=car2.drop(["Gender"],axis=1)
car2

car2.isna().sum() #no null values
car2.duplicated().sum()    #20 


'''
car3=car2.drop_duplicates()
car3

car3.isna().sum()             #no null values
car3.duplicated().sum()           #no duplicates
'''
#I am not removing duplicate values, as it makes accuracy less.

car3=car2
car3.columns

#boxplot for Gender_Female
plt.boxplot(car3["Gender_Female"])
plt.title("boxplot for gender")
plt.show()

#boxplot for Gender_male
plt.boxplot(car3["Gender_male"])
plt.title("boxplot for gender")
plt.show()

#boxplot for Age
plt.boxplot(car3["Age"])
plt.title("boxplot of age")
plt.show()

#boxplot for Estimated Salary
plt.boxplot(car3["EstimatedSalary"])
plt.title("boxplot of estimated salary")
plt.show()

#boxplot for Purchased
plt.boxplot(car3["Purchased"])
plt.title("boxplot of Purchased")
plt.show()
#no outliers

#histogram
car3.columns

car3[['Age', 'EstimatedSalary', 'Purchased', 'Gender_Female', 'Gender_Male']].hist(figsize=(15,8))
plt.show()

#seperating data into independent and dependent variables
ind_car=car3.drop(["Purchased"],axis=1)
dep_car=car3["Purchased"]

#standardization
#I am doing standardization because doing normalizatin is affecting the accuracy and most of the
#variables are in normal distribution and univariate distribution.
scaler = StandardScaler()
scaled_ind_car = scaler.fit_transform(ind_car) 
print(scaled_ind_car)

############################################## train and test split
#converting into dataframe
X=pd.DataFrame(scaled_ind_car, columns=ind_car.columns)

y=dep_car

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

################################################ Bernoulli Naive Bayes
#initializing the bernoulli nb
bnb = BernoulliNB()

#fitting train data of independent and dependent variables into bernoulli nb model, to train the model
model = bnb.fit(X_train, y_train)

#giving test data of independent variables to model to predict the dependent variable
y_pred = model.predict(X_test)  
y_pred                          #predictions of 20% of test data

#checking predicted(y_pred) values and y_test values are same or not/how much it matched
accuracy_test = np.mean(y_pred == y_test)    
accuracy_test           #68.7% / 69% values of test and predictions are same/ accuracy of test data

#giving independent variables of train data to the model for prediction.
train_pred = model.predict(X_train) 
train_pred

#checking predicted and train data of dependent values are same or not/how much it matched
accuracy_train = np.mean(train_pred == y_train)   
accuracy_train          #73.4% matched / accuracy of train data
#train and test accuracies are nearer. So no need of regularization technique.

############################################ confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], 
                                 index=['Predict Positive', 'Predict Negative'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu') 
#model predicted 55 values correctly and 25 values wrongly.

############################################ classification report
print(classification_report(y_test, y_pred))

TP = cm[1,1]
TN = cm[0,0]
FP = cm[1,0]
FN = cm[0,1]

#print classification accuracy
classification_accuracy = (TP + TN) / (TP + TN + FP + FN)   
print(classification_accuracy)        #65.7% values it predicted correctly out of 100%

# print classification error
classification_error = (FP + FN) / (TP + TN + FP + FN)       
print(classification_error)           #34.2% values it predicted wrongly out of 100%

############################################ checking which values it is taking more
# importing the collections module
import collections
# intializing the arr
arr = y_pred
# getting the elements frequencies using Counter class
elements_count = collections.Counter(arr)
# printing the element and the frequency
for key, value in elements_count.items():
   print(f"{key}: {value}")                      


















