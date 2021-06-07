"""## Importing the dataset"""

dataset = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_test = data_test.drop('id', axis=1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

print(data_test.head())

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
data_test = sc.fit_transform(data_test)

print(X_train)

print(X_test)

"""## Training the Logistic Regression model on the Training set"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""## Training the K-NN model on the Training set"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""# Training the SVM model on the Training set"""

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""# Training the Kernel SVM model on the Training set"""

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""# Training the Naive Bayes model on the Training set"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""# Training the Decision Tree Classification model on the Training set"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)

"""# Training the Random Forest Classification model on the Training set"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# predicting test dataset"""

test_pred = classifier.predict(data_test)
print(test_pred)
