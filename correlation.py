import pandas as pd

dataset = pd.read_csv('train.csv')

dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )

print("correlation score for supervised and sex is: ", dataset['Survived'].corr(dataset['Sex']))