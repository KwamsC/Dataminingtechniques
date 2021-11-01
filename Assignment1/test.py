import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class_data_path = './dataset.csv'
class_data = pd.read_csv(class_data_path, delimiter = ';')
pd.set_option('display.max_rows', 276)
pd.set_option('display.max_columns', 276)
# class_data = class_data.dropna(axis=0)

# class_data.columns = [c.replace(' ', '_') for c in class_data.columns]

y=class_data[class_data.columns[-1]]

# class_data = class_data.drop(['Timestamp', class_data.columns[-1]], axis=1)

v=class_data[class_data.columns[1]]

print(class_data.describe())

# print(class_data.info())


# print(v=='unknown'].sum())

# print(class_data.info())

# print(class_data.tail())
# print(list(class_data))


# class_features = ['What programme are you in?', 'Have you taken a course on machine learning?']
# X = class_data[class_features]

# print(X.head())
