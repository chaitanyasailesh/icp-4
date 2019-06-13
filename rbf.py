from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd



data_set=pd.read_csv("glass.csv")



x = data_set.drop('Type', axis=1)

y = data_set['Type']
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)



clf = SVC(kernel='rbf', C=1, gamma=0.1).fit(x_train, y_train)



y_pred = clf.predict(x_test)




print("Classification report is: \n", classification_report(y_test, y_pred))
print("Confusion matrix is: \n", confusion_matrix(y_test, y_pred))
print("Accuracy score is: ", accuracy_score(y_test, y_pred))