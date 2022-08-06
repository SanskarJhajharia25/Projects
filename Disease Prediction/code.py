#import statements
import pandas as pd
import io
import imageio
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_curve, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2,copy
import os, skimage,imageio,math
from scipy import ndimage
from skimage import io
from google.colab.patches import cv2_imshow
from IPython.display import display, Image
from skimage.io import imread_collection,imread
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#DIABETES Analysis
#The Diabetes Dataset is only meant to test the working of the ML models in python 
#We have trained the same on various ML models and compared them consequently to test the precision anmd accuracy of them
#Reading Data
diabetes_data = pd.read_csv( 'diabetes.csv')
diabetes_data.head()

#initial plots for better visual understanding of the various data available
%matplotlib inline

import matplotlib.pyplot as plt
fig, axes = plt.subplots( nrows=4, ncols=2 )
( ageHist, pregHist, glucoseHist, bldPressHist, triThickHist, insulHist, bmiHist, pedigrHist ) = axes.flatten()
fig.set_size_inches( 16, 9 )

ageHist.hist( diabetes_data[ "Age" ] )
ageHist.set_title( 'Age freq' )

pregHist.hist( diabetes_data[ "Pregnancies" ] )
pregHist.set_title( 'Pregnancy freq' )

glucoseHist.hist( diabetes_data[ "PlasmaGlucose" ] )
glucoseHist.set_title( 'Glucose freq' )

bldPressHist.hist( diabetes_data[ "DiastolicBloodPressure" ] )
bldPressHist.set_title( 'Blood pressure freq' )

triThickHist.hist( diabetes_data[ "TricepsThickness" ] )
triThickHist.set_title( 'Triceps freq' )

insulHist.hist( diabetes_data[ "SerumInsulin" ] )
insulHist.set_title( 'Insulin freq' )

bmiHist.hist( diabetes_data[ "BMI" ] )
bmiHist.set_title( 'BMI freq' )

pedigrHist.hist( diabetes_data[ "DiabetesPedigree" ] )
pedigrHist.set_title( 'Pedigree freq' )

plt.tight_layout()
plt.show()

#Using log function to normalise the skwed fields
diabetes_data = diabetes_data.assign( log_Age = lambda x: np.log( x[ 'Age' ] ) )

diabetes_data = diabetes_data.assign( zscore_glucose = zscore( diabetes_data[ 'PlasmaGlucose' ] ) )
diabetes_data = diabetes_data.assign( zscore_pressure = zscore( diabetes_data[ 'DiastolicBloodPressure' ] ) )
diabetes_data = diabetes_data.assign( zscore_thick = zscore( diabetes_data[ 'TricepsThickness' ] ) )
diabetes_data = diabetes_data.assign( zscore_insulin = zscore( diabetes_data[ 'SerumInsulin' ] ) )
diabetes_data = diabetes_data.assign( zscore_bmi = zscore( diabetes_data[ 'BMI' ] ) )

scaler = MinMaxScaler()

minMaxData = pd.DataFrame( scaler.fit_transform( diabetes_data.loc[ :, [ 'Pregnancies','DiabetesPedigree' ] ] ), columns = [ 'minMaxPreg', 'minMaxPedigree' ] )
diabetes_data = pd.concat( [ diabetes_data, minMaxData ], axis = 1, join = 'inner' )
diabetes_data.head()

#Training and Testing Data
#Using all features currently
train, test = train_test_split( diabetes_data, test_size = 0.3 )
features = [ "log_Age", "zscore_glucose", "zscore_pressure", "zscore_thick", "zscore_insulin", "zscore_bmi", "minMaxPreg", "minMaxPedigree" ]
X_train = train[ features ]
Y_train = train[ "Diabetic" ]
X_test = test[ features ]
Y_test = test[ "Diabetic" ]

bdt = AdaBoostClassifier( DecisionTreeClassifier( max_depth = 4 ) , algorithm="SAMME", n_estimators=200 )
dt = bdt.fit( X_train, Y_train )
Y_pred = dt.predict( X_test )
Y_probas = dt.predict_proba( X_test )
# score the model on the test data

#This function is meant to compare Score the model based on the test data (from the original dataset) and the predicted data
def scoreModel( Y_test, Y_pred ):
    # show accuracy, precision and recall
    from sklearn.metrics import accuracy_score
    score = accuracy_score( Y_test, Y_pred )
    print( "Accuracy: %.3f " % round( score, 3 ) )
    
    from sklearn.metrics import precision_score
    precScore = precision_score( Y_test, Y_pred, average = 'binary' )
    print( "Precision: %.3f " % round( precScore, 3 ) )
    
scoreModel( Y_test, Y_pred )

#XGBClassifier
from xgboost import XGBClassifier
XGB_clf =  XGBClassifier()
XGB_clf.fit(X_train, Y_train)
XGB_clf.score(X_test,Y_test)

#SVM
SVM_clf = svm.SVC()
SVM_clf.fit(X_train, Y_train)
SVM_clf.score(X_test,Y_test)

#KNN (n=10)
KNN_clf = KNeighborsClassifier(10)
KNN_clf.fit(X_train, Y_train)
KNN_clf.score(X_test,Y_test)

#MLP Classifier
MLP_clf =  MLPClassifier(alpha=1, max_iter=1000)
MLP_clf.fit(X_train, Y_train)
MLP_clf.score(X_test,Y_test)

#Random Forest
RF_clf =  RandomForestClassifier(max_depth=5, n_estimators=15, max_features=8)
RF_clf.fit(X_train, Y_train)
RF_clf.score(X_test,Y_test)

##################################################################################################

#HEART DISEASE
#Reading data
heart_data = pd.read_csv( 'heart.csv')
heart_data.head()
heart_data.shape

#Plotting the basic plots to visualise the data
%matplotlib inline
import matplotlib.pyplot as plt
fig, axes = plt.subplots( nrows=7, ncols=2 )
( ageHist, sexHist, chestPain, restingBP, cholestrolHist, fastBSHist , restingECGHist, maxHeart, exerciseAngina,oldpeakHist,slopeHist,CAHist, thalHist,target)=axes.flatten()
fig.set_size_inches( 20, 20 )


ageHist.hist( heart_data[ "age" ] )
ageHist.set_title( 'age freq' )


target.hist( heart_data[ "target" ] )
target.set_title( 'target variable' )

sexHist.hist(heart_data["sex"])
sexHist.set_title('Sex Var')

restingBP.hist(heart_data["trestbps"])
restingBP.set_title('trestbps frequency')

chestPain.hist(heart_data["cp"])
chestPain.set_title('chest pain frequency')

cholestrolHist.hist(heart_data["chol"])
cholestrolHist.set_title('chol frequency')

fastBSHist.hist(heart_data["fbs"])
fastBSHist.set_title('fbs frequency')

restingECGHist.hist(heart_data["restecg"])
restingECGHist.set_title('restecg frequency')

maxHeart.hist(heart_data["thalach"])
maxHeart.set_title('thalach frequency')

exerciseAngina.hist(heart_data["exang"])
exerciseAngina.set_title('exang frequency')

oldpeakHist.hist(heart_data["oldpeak"])
oldpeakHist.set_title('oldpeak frequency')

slopeHist.hist(heart_data["slope"])
slopeHist.set_title('slope frequency')

CAHist.hist(heart_data["ca"])
CAHist.set_title('ca frequency')

thalHist.hist(heart_data["thal"])
thalHist.set_title('thal frequency')

plt.tight_layout()
plt.show()

heart_data.describe().T.style.bar(subset=['mean'], color='#208ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')

sns.countplot(heart_data['target'])
print(pd.concat( [heart_data['target'].value_counts(), heart_data['target'].value_counts(normalize=True).mul(100).round(2)],
                 axis = 1,
                 keys = ('Count', 'Percentage')))

#Testing for all parameters
#Splitting into predictor and target variables
x_full = heart_data.drop('target', axis = 1)
y_full = heart_data['target']

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3, random_state = 1)

print('Train data records: %d \nTest data records: %d' % (x_train.shape[0], x_test.shape[0]))
logistic_model =LogisticRegression(max_iter = 1000)
logistic_model.fit(x_train, y_train)
train_predictions = logistic_model.predict(x_train)
test_predictions = logistic_model.predict(x_test)
model_performance = pd.DataFrame([[ 'Logistic Regression', 
                                    round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Train_Accuracy', 'Test_Accuracy'])
print(model_performance)

#Finding Correlation of each variable with the output (target)
plt.figure(figsize=(4, 6))
heatmap = sns.heatmap(heart_data.corr()[['target']].sort_values(by='target', ascending=False), 
                      vmin=-1, 
                      vmax=1, 
                      annot=True, 
                      cmap=sns.diverging_palette(5, 5, as_cmap=True))
heatmap.set_title('Features Correlating with Target', fontdict={'fontsize':18}, pad=16)

#Convert categorical variables into dummies
categorical_val = []
continous_val = []
for column in heart_data.columns:
    if len(heart_data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
categorical_val.remove('target')
dataset = pd.get_dummies(heart_data, columns = categorical_val)
dataset.head()

corr_data = heart_data.drop(['target'], axis = 1)
corrmat = corr_data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmin = -1, vmax= 1, square = True, annot = True,
            cmap=sns.diverging_palette(5, 5, as_cmap=True));

from sklearn.preprocessing import StandardScaler

#Basic transformations for the better fit of the variables
s_sc = StandardScaler()
col_to_scale = ['age', 'thalach', 'oldpeak', 'trestbps','chol']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

pd.set_option("display.float", "{:.2f}".format)
dataset.head()

dataset.describe().T.style.bar(subset=['mean'], color='#208ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')

# Splitting into predictor and target variables
x_full = dataset.drop('target', axis = 1)
y_full = dataset['target']

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size = 0.3, random_state = 42)

print('Train data records: %d \nTest data records: %d' % (x_train.shape[0], x_test.shape[0]))

logistic_model =LogisticRegression(max_iter = 1000)
logistic_model.fit(x_train, y_train)
train_predictions = logistic_model.predict(x_train)
test_predictions = logistic_model.predict(x_test)
mprf = pd.DataFrame([[ 'Logistic Regression', 
                                    round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
moddf=model_performance.append(mprf,ignore_index=True)
moddf

#XGB
from xgboost import XGBClassifier
XGB_clf =  XGBClassifier()
XGB_clf.fit(x_train, y_train)
train_predictions = XGB_clf.predict(x_train)
test_predictions = XGB_clf.predict(x_test)
mprff = pd.DataFrame([[ 'XGB Classifier', 
                                   round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
moddf=moddf.append(mprff,ignore_index=True)
moddf

#MLP
MLP_clf =  MLPClassifier(alpha=1, max_iter=1000)
MLP_clf.fit(x_train, y_train)
train_predictions = MLP_clf.predict(x_train)
test_predictions = MLP_clf.predict(x_test)
model_performance = pd.DataFrame([[ 'MLP Classifier', 
                                    round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
model_performance

#Random Forest Classifier
RF_clf =  RandomForestClassifier(max_depth=5, n_estimators=15, max_features=8)
RF_clf.fit(x_train, y_train)
train_predictions = RF_clf.predict(x_train)
test_predictions = RF_clf.predict(x_test)
mprfff = pd.DataFrame([[ 'RF Classifier', 
                                    round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
moddf=moddf.append(mprfff,ignore_index=True)
moddf


#SVM
SVM_clf = svm.SVC()
SVM_clf.fit(x_train, y_train)
train_predictions = SVM_clf.predict(x_train)
test_predictions = SVM_clf.predict(x_test)
mprffff = pd.DataFrame([[ 'SVM Classifier', 
                                    round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
moddf=moddf.append(mprffff,ignore_index=True)
moddf

#KNN (n=5)
KNN_clf = KNeighborsClassifier(5)
KNN_clf.fit(x_train, y_train)
train_predictions = KNN_clf.predict(x_train)
test_predictions = KNN_clf.predict(x_test)
mprfffff = pd.DataFrame([[ 'KNN Classifier', 
                                   round(precision_score(y_test, test_predictions)*100,2),
                                    round(recall_score(y_test, test_predictions)*100,2),
                                    round(f1_score(y_test, test_predictions)*100,2),round(accuracy_score(y_train, train_predictions)*100,2),
                                    round(accuracy_score(y_test, test_predictions)*100,2)]],
                                   columns = ['Model', 'Precision Score','Recall Score','F1 Score','Train_Accuracy', 'Test_Accuracy'])
moddf=moddf.append(mprfffff,ignore_index=True)
moddf

