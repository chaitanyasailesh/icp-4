from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd




data_set = pd.read_csv("glass.csv")



x = data_set.drop('Type', axis=1)
y = data_set['Type']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)



clf = SVC(kernel='linear', C=1).fit(x_train, y_train)



y_pred = clf.predict(x_test)



print("Confusion matrix is: \n", confusion_matrix(y_test, y_pred))
print("Classification report is: \n", classification_report(y_test, y_pred))
print("Accuracy score is: ", accuracy_score(y_test, y_pred))