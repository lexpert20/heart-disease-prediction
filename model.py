simplefilter(action='ignore', category=FutureWarning)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('cleveland.csv', header=None)

# label the dataset
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
# EDA
print("key are ", df.keys())
print("no of rows and columns ", df.shape)  # display number of rows and columns
print("targets ", df.target)  # display number of rows and columns
print("head \n", df.head())  # display number of rows and columns
print("info \n", df.info())  # display number of rows and columns
# profile = ProfileReport(df, title="Aboubacar & Wiame, heart data set Report")
# print(profile)
# profile.to_widgets()
# profile.to_file("report.html")

print("describe \n", df.describe())  # display number of rows and columns

### 1 == male, 0 == female
df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # mapping so that 0 means absent and 1,2,3,4 means present
df['sex'] = df.sex.map({0: 'female', 1: 'male'})

# Fill na with mean as they are not a lot of them
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

import matplotlib.pyplot as plt
import seaborn as sns

# distribution of target vs age
sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=df, x='age', hue='target', order=df['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

# barplot of age vs sex with hue = target
sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')

# another visual aspect
# sns.catplot(kind='point', data=df, y='age', x='sex', hue='target')

plt.title('Distribution of age vs sex with the target class')
plt.show()

df['sex'] = df.sex.map({'female': 0, 'male': 1})

################################## data preprocessing

'''X = df.drop('target',axis=1)  # all cols excepts target
y = df.iloc[:, -1].values  # target col'''

y = df["target"]
X = df.drop('target', axis=1)  # all cols excepts target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("X_test:\n", X_test.head(1))
y_test = y_test.head(1)
'''print("X_train", X_train)
print("y_train", y_train)
print("y_test", y_test)'''

from sklearn.preprocessing import StandardScaler as ss

# Getting X => 80% X training and remaining 20% X testing
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test.head(1))

#########################################   SVM   #############################################################
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')
# Train the model
classifier.fit(X_train, y_train)

# Predicting using X_test , y_pred is the values we get after testing using our 20% X_test
print("X_test:\n", X_test)
y_predicted_with_X_test = classifier.predict(X_test)
print("y predict \n", y_predicted_with_X_test)
print("y real \n", y_test)
from sklearn.metrics import confusion_matrix

'''Now we check, the result we got from our testing with what we should have gotten (y_test)
 it gives us the accuracy for the testing data'''
cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

# Try to predict based on the X_train instead of X_test
y_pred_with_X_train = classifier.predict(X_train)

'''Now we check, the result we got from our testing with X_train with what we should have gotten (y_test)
 it gives us the accuracy for the training data'''
cm_train = confusion_matrix(y_pred_with_X_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for svm = {}'.format(
    (cm_test[0][0] + cm_test[1][1]) / len(y_test)))  # This score is the ability to learn

#########################################   Naive Bayes  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predicted_with_X_test = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_with_X_train, y_train)

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################   Logistic Regression  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predicted_with_X_test = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_with_X_train, y_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################   Decision Tree  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predicted_with_X_test = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_with_X_train, y_train)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################  Random Forest  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predicted_with_X_test = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_with_X_train, y_train)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

###############################################################################
# applying lightGBM
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
# fine turning params
'''params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10'''

clf = lgb.train(params, d_train, 100)
# Prediction
y_predicted_with_X_test = clf.predict(X_test)
# convert into binary values
for i in range(0, len(y_predicted_with_X_test)):
    if y_predicted_with_X_test[i] >= 0.5:  # setting threshold to .5
        y_predicted_with_X_test[i] = 1
    else:
        y_predicted_with_X_test[i] = 0

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = clf.predict(X_train)

for i in range(0, len(y_pred_with_X_train)):
    if y_pred_with_X_train[i] >= 0.5:  # setting threshold to .5
        y_pred_with_X_train[i] = 1
    else:
        y_pred_with_X_train[i] = 0

cm_train = confusion_matrix(y_pred_with_X_train, y_train)
print()
print('Accuracy for training set for LightGBM = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for LightGBM = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

###############################################################################
# applying XGBoost

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier

xg = XGBClassifier()
xg.fit(X_train, y_train)
y_predicted_with_X_test = xg.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_predicted_with_X_test, y_test)

y_pred_with_X_train = xg.predict(X_train)

for i in range(0, len(y_pred_with_X_train)):
    if y_pred_with_X_train[i] >= 0.5:  # setting threshold to .5
        y_pred_with_X_train[i] = 1
    else:
        y_pred_with_X_train[i] = 0

cm_train = confusion_matrix(y_pred_with_X_train, y_train)
print()
print('Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))