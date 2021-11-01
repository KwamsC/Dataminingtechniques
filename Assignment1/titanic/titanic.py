import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
sns.set()
from sklearn.model_selection import cross_val_score
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

titanic_train = pd.read_csv('./train.csv')
titanic_test = pd.read_csv('./test.csv')

titanic_train_test = [titanic_train, titanic_test]

print("hello")

# # print(titanic_train.describe(include='all'))





# # titaninc_train['Price'].value_counts()

# pd.set_option('display.max_rows', 276)
# pd.set_option('display.max_columns', 276)

# for dataset in titanic_train_test:
#     dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)   
#     # dataset['Age class'] = dataset['Age'].map(lambda x: 0 if x < 18 else (1 if x>=18 and x<36 else 2))


# print(dataset['Title'])
# #Mr: 0
# #Miss: 1
# #Mrs : 2
# #Others: 3

# title_mapping = {"Mr":0, "Miss": 1, "Mrs": 2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Mlle":3,
# "Major":3, "Don":3, "Mme":3, "Ms":3, "Jonkheer":3, "Sir":3, "Lady":3, "Countess":3, "Capt":3 }

# embarked_mapping = {'S':0, 'C':1, 'Q':2}

# sex_mapping = {'female':0, 'male':1}

# to_drop = ['Name', 'Cabin', 'Ticket' ]
# # titanic_train_test = titanic_train_test.drop(to_drop, axis=1)
# # titanic_test = titanic_test.drop(to_drop, axis=1)

# for dataset in titanic_train_test:
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# for dataset in titanic_train_test:
#     dataset['Age'].fillna(dataset.groupby("Title")['Age'].transform("median"), inplace=True)

# # facet = sns.FacetGrid(titanic_train, hue = "Survived", aspect=4)
# # facet.map(sns.kdeplot, 'Age', shade=True)
# # facet.set(xlim=(0, titanic_train['Age'].max()))
# # facet.add_legend()
# # plt.show()


# # def bar_chart(feature):
# #     survived = titanic_train[titanic_train['Survived']==1][feature].value_counts()
# #     dead = titanic_train[titanic_train['Survived']==0][feature].value_counts()
# #     df = pd.DataFrame([survived, dead])
# #     df.index = ['Survived', 'Dead']
# #     df.plot(kind='bar', stacked=True, figsize=(10,5))
# #     plt.show(block=True)

# for dataset in titanic_train_test:
#     dataset.loc[dataset['Age'] <= 16, 'Age']=0,
#     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
#     dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
#     dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
#     dataset.loc[(dataset['Age'] > 62), 'Age']=4,

# #Embarked
# # Pclass1 = titanic_train[titanic_train['Pclass']==1]['Embarked'].value_counts()
# # Pclass2 = titanic_train[titanic_train['Pclass']==2]['Embarked'].value_counts()
# # Pclass3 = titanic_train[titanic_train['Pclass']==3]['Embarked'].value_counts()
# # df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
# # df.index = ['1st class', '2nd class', '3rd class' ]
# # df.plot(kind='bar', stacked = True, figsize=(10,55))
# # plt.show(block=True)

# # bar_chart('Embarked')


# for dataset in titanic_train_test:
#     dataset['Embarked'] = dataset['Embarked'].fillna('S')

# for dataset in titanic_train_test:
#     dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# # def age_class(age):
# #     if(age < 18):
# #         return 'child'
# #     elif(age > 65):
# #         return 'senior'
# #     else:
# #         return 'adult'

# # titanic_train['Age class'] = titanic.apply(lambda row: age_class(row['Age']), axis=1)

# titanic_train['Pclass'].dropna().hist(bins=16, range=(1,3), alpha=.5)
# # plt.show()

# # print(titanic_train['Age'].value_counts())

# # bar_chart('Parch')

# # print(titanic_train_test)

# # titanic_data.describe()

# # print(titanic_train.describe())
# # print(titanic_test.describe())



# # print(titanic_data.columns.values)

# # titanic_features = ['Sex', 'Age']

# to_drop = ['Name', 'Cabin', 'Ticket', 'Fare','PassengerId' ]
# titanic_train = titanic_train.drop(to_drop, axis=1)

# to_drop = ['Name', 'Cabin', 'Ticket', 'Fare', 'PassengerId']
# titanic_test = titanic_test.drop(to_drop, axis=1)

# # bar_chart('Fare')


# # print(titanic_train.head(20))

# # print(titanic_test.head(20))

# print(titanic_train.info())
# print(titanic_test.info())

# target = titanic_train.Survived

# clf SVC()
# clf.fit(titanic_train, target)


# # print(y)
