import pydotplus
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
from sklearn.externals.six import StringIO
from sklearn import tree

with open('lenses.txt', 'r') as fr:
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_target = []
for each in lenses:
    lenses_target.append(each[-1])

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses_list = []
lenses_dict = {}
for each_label in lensesLabels:
    for each in lenses:
        lenses_list.append(each[lensesLabels.index(each_label)])
    lenses_dict[each_label] = lenses_list
    lenses_list = []
#print(lenses_dict)
lenses_pd = pd.DataFrame(lenses_dict)
print(lenses_pd)
le = LabelEncoder()
for col in lenses_pd.columns:
    lenses_pd[col] = le.fit_transform(lenses_pd[col])
print(lenses_pd)
clf = tree.DecisionTreeClassifier()
lenses = clf.fit(lenses_pd, lenses_target)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=lenses_pd.keys(),
                     class_names=clf.classes_,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('tree.pdf')
print(clf.predict([[1,1,1,0]]))