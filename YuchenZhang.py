import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
import warnings
warnings.filterwarnings('ignore')




df = pd.read_csv('data.csv')


attri_y = df.loc[df['Attrition'] == 'Yes']
attri_n = df.loc[df['Attrition'] == 'No']

def transFormat(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset

y_n_dic={"Yes":1,"No":0}
df['Attrition'] = transFormat(df['Attrition'],y_n_dic)

BusinessTravel_dic = {"Travel_Rarely":1,"Travel_Frequently":2, "Non-Travel":3}
Department_dic = {"Research & Development":1, "Sales":2, "Human Resources":3}
EducationField_dic = {"Life Sciences":1, "Medical":2, "Marketing":3, "Technical Degree":4, "Human Resources":5, "Other":6 }
Gender_dic = {"Female":1, "Male":2}
JobRole_dic = {"Sales Executive":1, "Research Scientist":2, "Laboratory Technician":3, "Manufacturing Director":4, "Manager":5, "Research Director":6, "Human Resources":7, "Sales Representative":8, "Healthcare Representative":9}
df['BusinessTravel'] = transFormat(df['BusinessTravel'], BusinessTravel_dic)
df['Department'] = transFormat(df['Department'], Department_dic)
df['EducationField'] = transFormat(df['EducationField'], EducationField_dic)
df['Gender'] = transFormat(df['Gender'], Gender_dic)
df['JobRole'] = transFormat(df['JobRole'], JobRole_dic)
MaritalStatus_dic = {'Married':1,'Single':2, 'Divorced':3}
df['MaritalStatus'] = transFormat(df['MaritalStatus'],MaritalStatus_dic)
OverTime_dic = {'Yes':1,'No':0}
df['OverTime'] = transFormat(df['OverTime'],OverTime_dic)
df = df.drop(columns = 'Over18')

attri_p = df.loc[df['Attrition'] == 1]
attri_n = df.loc[df['Attrition'] == 0]
label_p = attri_p['Attrition']
label_n = attri_n['Attrition']

# distribution
data_set_yes = df[df['Attrition'] == 1]
data_set_no = df[df['Attrition'] == 0]


def boxplotForY_n(feature, i, j,data_yes = data_set_yes, data_no = data_set_no):
    feature_y = data_set_yes[feature]
    feature_n = data_set_no[feature]
    axs[i, j].boxplot([feature_y.values, feature_n.values], patch_artist=True, labels=['Yes', 'No'])
    axs[i, j].set_title(feature)

fig, axs = plt.subplots(3, 3)
boxplotForY_n('Age',0,0)
boxplotForY_n('DailyRate',0,1)
boxplotForY_n('DistanceFromHome',0,2)
boxplotForY_n('EmployeeCount',1,0)
boxplotForY_n('EmployeeNumber',1,1)
boxplotForY_n('HourlyRate',1,2)
boxplotForY_n('MonthlyIncome',2,0)
boxplotForY_n('MonthlyRate',2,1)
boxplotForY_n('NumCompaniesWorked',2,2)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
fig.savefig('boxplot1.png')

fig, axs = plt.subplots(3, 3)
boxplotForY_n('PercentSalaryHike',0,0)
boxplotForY_n('StandardHours',0,1)
boxplotForY_n('TotalWorkingYears',0,2)
boxplotForY_n('TrainingTimesLastYear',1,0)
boxplotForY_n('YearsAtCompany',1,1)
boxplotForY_n('YearsInCurrentRole',1,2)
boxplotForY_n('YearsSinceLastPromotion',2,0)
boxplotForY_n('YearsWithCurrManager',2,1)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
fig.savefig('boxplot2.png')

attri_p = attri_p.drop(columns='EmployeeCount')
attri_n = attri_n.drop(columns='EmployeeCount')
attri_p = attri_p.drop(columns='StandardHours')
attri_n = attri_n.drop(columns='StandardHours')

attri_p = attri_p.drop(columns='Attrition')
attri_n = attri_n.drop(columns='Attrition')



x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(attri_p,label_p,test_size = 50)
x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(attri_n,label_n,test_size = 50)

# get test set
x_test = x_test_p.values.tolist() + x_test_n.values.tolist()
y_test = y_test_p.values.tolist() + y_test_n.values.tolist()
index = list(np.arange(len(x_test)))
np.random.shuffle(index)
x_test_rdm=[]
for x in index:
    x_test_rdm.append(x_test[x])
y_test_rdm = []
for x in index:
    y_test_rdm.append(y_test[x])

def overSampling(x_train_p, y_train_p, x_train_n, y_train_n, size):
    x = []
    y = []
    index = list(np.arange(len(x_train_p)))
    for i in range(size):
        cur = np.random.choice(index)
        x.append(x_train_p.iloc[cur])
        y.append(y_train_p.iloc[cur])

    index = list(np.arange(len(x_train_n)))
    for i in range(size):
        cur = np.random.choice(index)
        x.append(x_train_n.iloc[cur])
        y.append(y_train_n.iloc[cur])
    # shuffle
    index = list(np.arange(size * 2))
    np.random.shuffle(index)
    x_train_rdm = []
    for i in index:
        x_train_rdm.append(x[i])
    y_train_rdm = []
    for i in index:
        y_train_rdm.append(y[i])

    return x_train_rdm, y_train_rdm


def genTrainSet(x_train_p, y_train_p, x_train_n, y_train_n, size):
    index = list(np.arange(len(x_train_p)))
    x = []
    y = []
    for i in range(size):
        cur = np.random.choice(index)
        #         print(cur)
        #         print(x_train_p.shape)
        #         print(x_train_p)
        #         print(x_train_p.iloc[cur])
        x.append(x_train_p.iloc[cur])
        y.append(y_train_p.iloc[cur])

    index = list(np.arange(len(x_train_n)))
    for i in range(size):
        cur = np.random.choice(index)
        x.append(x_train_n.iloc[cur])
        y.append(y_train_n.iloc[cur])

    index = list(np.arange(size * 2))
    np.random.shuffle(index)
    x_train_rdm = []
    for i in index:
        x_train_rdm.append(x[i])
    y_train_rdm = []
    for i in index:
        y_train_rdm.append(y[i])

    return x_train_rdm, y_train_rdm


# mix all training data together
def mix_data_direct(x_train_p,y_train_p,x_train_n,y_train_n):
    x_train = x_train_p.values.tolist() + x_train_n.values.tolist()
    y_train = y_train_p.values.tolist() + y_train_n.values.tolist()


    index = list(np.arange(len(x_train)))
    np.random.shuffle(index)
    x_train_rdm=[]
    for x in index:
        x_train_rdm.append(x_train[x])
    y_train_rdm = []
    for x in index:
        y_train_rdm.append(y_train[x])
    return x_train_rdm, y_train_rdm


# evaluate
def evaluate(title, y_true, y_pred):
    print(title)
    print("Accuracy score: {}".format(accuracy_score(y_true, y_pred)))
    print("=" * 80)
    print(classification_report(y_true, y_pred))

###### decision tree
import pydotplus

from sklearn import tree
import collections

data_feature_names = attri_p.columns

# Visualize data
def visulizeTree(dctree, file_name, feature_names = data_feature_names):
    dot_data = tree.export_graphviz(dctree,
                                    feature_names=feature_names,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(file_name)

# training data
cur_train_x, cur_train_y = genTrainSet(x_train_p,y_train_p,x_train_n,y_train_n,50)
x_train_org, y_train_org = mix_data_direct(x_train_p, y_train_p, x_train_n, y_train_n)
x_train_os, y_train_os = overSampling(x_train_p, y_train_p, x_train_n, y_train_n, len(x_train_n))


print('Decision Tree')
# using balanced samples
clf = tree.DecisionTreeClassifier()

clf.fit(cur_train_x, cur_train_y)
visulizeTree(clf,file_name='balanced_tree.png')
y_pred = clf.predict(x_test_rdm)
evaluate("DT-100-sample", y_test_rdm, y_pred)


# using imbalanced samples
clf_imb = tree.DecisionTreeClassifier()
clf_imb.fit(x_train_org, y_train_org)
visulizeTree(clf_imb,file_name='imbalanced_tree.png')
y_pred = clf_imb.predict(x_test_rdm)
evaluate("DT-diret", y_test_rdm, y_pred)

# using oversampling samples
clf_os = tree.DecisionTreeClassifier()
clf_os.fit(x_train_os, y_train_os)
visulizeTree(clf_os,file_name='oversampling_tree.png')
y_pred = clf_os.predict(x_test_rdm)
evaluate("DT-2000-sample", y_test_rdm, y_pred)


# Naive Bayes
# using orignial
gnb1 = GaussianNB()
gnb1.fit(x_train_org, y_train_org)
y_pred = gnb1.predict(x_test_rdm)
evaluate("NB-original", y_test_rdm, y_pred)

# using oversampling
gnb2 = GaussianNB()
gnb2.fit(x_train_os, y_train_os)
y_pred = gnb2.predict(x_test_rdm)
evaluate("NB-oversampling", y_test_rdm, y_pred)

def randomforest(n_estimators_=1000, max_depth_=4,max_features_=0.3, min_samples_split_=2):
    forest = RandomForestClassifier(n_estimators=n_estimators_, max_depth=max_depth_, max_features=max_features_, min_samples_split=min_samples_split_, bootstrap=True,
                                    n_jobs=3)
    return forest

# Random Forest
# using original data
fst1 = randomforest()
fst1.fit(x_train_org, y_train_org)
y_pred = fst1.predict(x_test_rdm)
evaluate("RF -original", y_test_rdm, y_pred)

# using oversampling data
fst2 = randomforest()
fst2.fit(x_train_os, y_train_os)
y_pred = fst2.predict(x_test_rdm)
evaluate("RF-oversample", y_test_rdm, y_pred)

# LR
# using orignal
lr1 = LogisticRegression()
lr1.fit(x_train_org, y_train_org)
y_pred = lr1.predict(x_test_rdm)
evaluate("LR-original", y_test_rdm, y_pred)

# using oversampling
lr2 = LogisticRegression()
lr2.fit(x_train_os, y_train_os)
y_pred = lr2.predict(x_test_rdm)
evaluate("LR-oversample", y_test_rdm, y_pred)

# using batch trainging
lr3 = LogisticRegression()
for i in range(1000):
    x_cur, y_cur = genTrainSet(x_train_p, y_train_p, x_train_n, y_train_n, 50)
    lr3.fit(x_cur, y_cur)
y_pred = lr3.predict(x_test_rdm)
evaluate("LR-batch", y_test_rdm, y_pred)
