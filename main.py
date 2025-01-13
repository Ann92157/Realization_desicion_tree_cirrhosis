import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score
from DecisionTreeModel import Node, Desicion_tree 

warnings.filterwarnings("ignore")
df = pd.read_csv("cirrhosis.csv")
df['Drug'].replace({'D-penicillamine':0, 'Placebo':1}, inplace=True)
df['Sex'].replace({'F':0, 'M':1}, inplace=True)
df['Ascites'].replace({'N':0, 'Y':1}, inplace=True)
df['Hepatomegaly'].replace({'N':0, 'Y':1}, inplace=True)
df['Spiders'].replace({'N':0, 'Y':1}, inplace=True)
df['Edema'].replace({'N':0, 'S':1, 'Y':2}, inplace=True)


df = df.dropna()
df = df.drop(columns = 'ID')
X = df.drop(columns = ['Status', 'N_Days', 'Stage'])
y = df['Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
features = list(X_train.columns)
classifier = Desicion_tree(min_samples_split=2, max_depth=3)
classifier.fit(X_train,y_train)

Y_pred = classifier.predict(X_test) 
classifier.print_tree(classifier.root, features)
print(f"Accuracy: {accuracy_score(y_test, Y_pred)}")
