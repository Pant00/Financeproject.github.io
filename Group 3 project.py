#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### load the file
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# In[2]:


df=pd.read_csv("P:/Project/finance_train.csv")
print(df.shape)


# In[3]:


df.drop(columns=['REF_NO'],inplace=True)
df.shape


# In[4]:


import re
# Working on Region Column
postcode_regex = re.compile(r'^([A-Z]{1,2})')


# In[5]:


df['newpostarea']=df['post_area'].apply(lambda x: postcode_regex.match(x).group(1) if postcode_regex.match(x) else None)


# In[6]:


df_ref=pd.read_csv("P:\Project\post_code.csv")


# In[7]:


post_code_to_region=dict(zip(df_ref['post_code'], df_ref['region']))


# In[8]:


df['new_region']=df['newpostarea'].map(post_code_to_region)


# In[9]:


df.drop(columns=['newpostarea','region'],inplace=True)
df.shape


# In[10]:


df.head()


# In[11]:


df['family_income'].unique()


# In[12]:


categroical_columns=[]
for col in df.columns:
    if df[col].dtype=='object':
        categroical_columns.append(col)
print(categroical_columns)


# In[13]:


categroical_columns.remove('gender')
for col in categroical_columns:
    df[col] = df[col].replace(['Unknown', 'Unclassified'], np.nan)


# In[14]:


df.isnull().sum()


# In[15]:


df['children'].replace({'1':'One','2':'two','3':'three','4+':'more than 3'},inplace=True)


# In[16]:


df['family_income'].isnull().sum()
df['family_income'].fillna(df['family_income'].mode()[0],inplace=True)
df['family_income'] = df['family_income'].astype(str)


# In[17]:


df['family_income'].isnull().sum()


# In[18]:


def extract_salary_range(salary_range):
    pattern = r"([\d,]+)" # Match any sequence of digits and commas
    matches = re.findall(pattern, salary_range)
    return [int(match.replace(",", "")) for match in matches]


# In[19]:


df['family_income']=df['family_income'].apply(extract_salary_range)


# In[20]:


df['family_min_income'] = df['family_income'].str[0]
df['family_max_income'] = df['family_income'].str[1]
print("Null Values in Family_max_column",df['family_max_income'].isnull().sum())
print("Null values in family min column",df['family_min_income'].isnull().sum())
df['family_max_income']=df['family_max_income'].fillna(df['family_min_income'])
df['family_avg_income']=(df['family_max_income']+df['family_min_income'])/2


# In[21]:


df['family_avg_income'].isnull().sum()


# In[22]:


df.drop(columns=['family_income','family_max_income','family_min_income'],inplace=True)


# In[23]:


df=df[df['year_last_moved']!=0]


# In[24]:


df['year_last_moved']=pd.to_datetime(df['year_last_moved'], format='%Y')
df['year_last_moved'] = df['year_last_moved'].dt.year
df.shape


# In[25]:


df['loan']=df['Home.Loan']+df['Personal.Loan']


# In[26]:


df.drop(columns=['Home.Loan','Personal.Loan'],inplace=True)


# In[27]:


df['total_insurance']=df['Life.Insurance']+df['Medical.Insurance']
df.drop(columns=['Life.Insurance','Medical.Insurance'],inplace=True)


# In[28]:


df['total_investment']=df['Investment.in.Commudity']+df['Investment.in.Equity']+df['Investment.in.Derivative']
df.drop(columns=['Investment.in.Commudity','Investment.in.Equity','Investment.in.Derivative'],inplace=True)
df.shape


# In[29]:


df['family_avg_income'].unique()


# In[30]:


df['age_band'].unique()


# In[31]:


# remove 0 values from last_year_moved and change the column into year
df=df[df['year_last_moved']!=0]
# changing data type to year
df['year_last_moved']=pd.to_datetime(df['year_last_moved'], format='%Y')
df['year_last_moved'] = df['year_last_moved'].dt.year
df.shape


# In[32]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.20,random_state=1)
print(train.shape)
print(test.shape)


# In[33]:


train['Source']='Train'
test['Source']='Test'
print(train.shape)
print(test.shape)


# In[34]:


train['Revenue.Grid'].value_counts()


# In[35]:


children_max=train[train['age_band'].isnull()]['children'].value_counts().idxmax()
print(children_max)
# filtering the data which do not have null values
new_df=train[~pd.isna(train['age_band'])]
tofill=new_df[(new_df['children']==children_max)]['age_band'].value_counts().idxmax()
print(tofill)


# In[36]:


train['age_band'].fillna(tofill,inplace=True)
test['age_band'].fillna(tofill,inplace=True)


# In[37]:


age_band_max=train[train['status'].isnull()]['age_band'].value_counts().idxmax()
children_max=train[train['status'].isnull()]['children'].value_counts().idxmax()
print(age_band_max)
print(children_max)


# In[38]:


new_df=train[~pd.isna(train['status'])]
tofill=train[(train['age_band']==age_band_max) & (train['children']==children_max)]['status'].value_counts().idxmax()
print(tofill)


# In[39]:


train['status'].fillna(tofill,inplace=True)
test['status'].fillna(tofill,inplace=True)


# In[40]:


age_band_max=train[train['occupation'].isnull()]['age_band'].value_counts().idxmax()
children_max=train[train['occupation'].isnull()]['children'].value_counts().idxmax()
status_max=train[train['occupation'].isnull()]['status'].value_counts().idxmax()
print(age_band_max)
print(children_max)
print(status_max)


# In[41]:


new_df=train[~pd.isna(train['occupation'])]
tofill=train[(train['age_band']==age_band_max) & (train['children']==children_max) & (train['status']==status_max)]['occupation'].value_counts().idxmax()
print(tofill)


# In[42]:


train['occupation'].fillna(tofill,inplace=True)
test['occupation'].fillna(tofill,inplace=True)


# In[43]:


train['occupation'].isnull().sum()


# In[44]:


age_band_max=train[train['occupation_partner'].isnull()]['age_band'].value_counts().idxmax()
children_max=train[train['occupation_partner'].isnull()]['children'].value_counts().idxmax()
status_max=train[train['occupation_partner'].isnull()]['status'].value_counts().idxmax()
occupation_max=train[train['occupation_partner'].isnull()]['occupation'].value_counts().idxmax()
print(age_band_max)
print(children_max)
print(status_max)
print(occupation_max)




# In[45]:


new_df=train[~pd.isna(train['occupation_partner'])]
tofill=train[(train['age_band']==age_band_max) & (train['children']==children_max) & (train['status']==status_max) & (train['occupation']==occupation_max)]['occupation_partner'].value_counts().idxmax()
print(tofill)


# In[46]:


train['occupation_partner'].fillna(tofill,inplace=True)
test['occupation_partner'].fillna(tofill,inplace=True)


# In[47]:


age_band_max=train[train['home_status'].isnull()]['age_band'].value_counts().idxmax()
children_max=train[train['home_status'].isnull()]['children'].value_counts().idxmax()
status_max=train[train['home_status'].isnull()]['status'].value_counts().idxmax()
occupation_max=train[train['home_status'].isnull()]['occupation'].value_counts().idxmax()
partner_occupaion=train[train['home_status'].isnull()]['occupation_partner'].value_counts().idxmax()
print(age_band_max)
print(children_max)
print(status_max)
print(occupation_max)
print(partner_occupaion)


# In[48]:


new_df=train[~pd.isna(train['home_status'])]
tofill=train[(train['age_band']==age_band_max) & (train['children']==children_max) & (train['status']==status_max) & (train['occupation']==occupation_max)]['home_status'].value_counts().idxmax()
print(tofill)


# In[49]:


train['home_status'].fillna(tofill,inplace=True)
test['home_status'].fillna(tofill,inplace=True)


# In[50]:


region_max=train[train['TVarea'].isnull()]['new_region'].value_counts().idxmax()
print(region_max)


# In[51]:


new_df=train[~pd.isna(train['TVarea'])]
tofill=train[train['new_region']==region_max]['TVarea'].value_counts().idxmax()
print(tofill)


# In[52]:


train['TVarea'].fillna(tofill,inplace=True)
test['TVarea'].fillna(tofill,inplace=True)


# In[53]:


fullraw=pd.concat([train,test],axis=0)
fullraw.shape


# In[54]:


from sklearn.preprocessing import LabelEncoder


# In[55]:


lab = LabelEncoder()


# In[56]:


fullraw['age_band2'] = lab.fit_transform(fullraw['age_band'])


# In[57]:


fullraw['age_band'].unique()


# In[58]:


fullraw['age_band2'].unique()


# In[59]:


fullraw.head()


# In[60]:


fullraw.drop(columns=["age_band"])


# In[61]:


from category_encoders import TargetEncoder


# In[62]:


le=TargetEncoder()
le.fit(fullraw["TVarea"],fullraw["Revenue.Grid"])
values=le.transform(fullraw["TVarea"])


# In[63]:


fullraw.drop(columns=['TVarea'],inplace=True)


# In[64]:


fullraw=pd.concat([fullraw,values],axis=1)


# In[65]:


fullraw.head()


# In[66]:
pip install category_encoders
import category_encoders as ce

le=TargetEncoder()
le.fit(fullraw["post_code"],fullraw["Revenue.Grid"])
values2=le.transform(fullraw["post_code"])


# In[67]:


fullraw.drop(columns=['post_code'],inplace=True)


# In[68]:


fullraw=pd.concat([fullraw,values2],axis=1)


# In[69]:


fullraw.head()


# In[74]:


for col in fullraw.columns:
    print(col,":",len(fullraw[col].unique()),"labels")


# In[76]:


fullraw.shape


# In[77]:


fullRaw2=pd.get_dummies(fullraw,drop_first=True)
fullRaw2.shape


# In[80]:


fullraw.drop(columns=["post_area"],inplace=True)


# In[81]:


for col in fullraw.columns:
    print(col,":",len(fullraw[col].unique()),"labels")


# In[82]:


fullraw["year_last_moved"].dtypes


# In[83]:


fullRaw3=pd.get_dummies(fullraw,drop_first=True)
fullRaw3.shape


# In[84]:


fullraw.drop(columns=["age_band"],inplace=True)


# In[85]:


fullRaw4=pd.get_dummies(fullraw,drop_first=True)
fullRaw4.shape


# In[86]:


Train = fullRaw4[fullRaw4['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
Test = fullRaw4[fullRaw4['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()
print(Train.shape)
print(Test.shape)


# In[87]:


depVar = "Revenue.Grid"
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()


# In[88]:


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[89]:


from sklearn.ensemble import RandomForestClassifier
RRR=RandomForestClassifier(n_estimators=200)
RRR.fit(trainX,trainY)


# In[90]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("train",accuracy_score(trainY,RRR.predict(trainX)))
print("test",accuracy_score(testY,RRR.predict(testX)))
print(classification_report(trainY, RRR.predict(trainX)))
print(classification_report(testY,RRR.predict(testX)))
print("Confusion matrix:")
print(confusion_matrix(trainY,RRR.predict(trainX)))
print(confusion_matrix(testY,RRR.predict(testX)))


# In[91]:


importance=RRR.feature_importances_
final_df=pd.DataFrame({"Features":trainX.columns,"importances":importance})
final_df.set_index('importances')
final_df=final_df.sort_values('importances',ascending=False).head(15)


# In[92]:


final_df


# In[97]:


new=fullraw[["Online.Purchase.Amount","Investment.Tax.Saving.Bond","total_investment","Average.Credit.Card.Transaction","total_insurance","Portfolio.Balance","loan","Term.Deposit","Average.A.C.Balance","Investment.in.Mutual.Fund","Balance.Transfer","year_last_moved","age_band2","Revenue.Grid","Source"]]


# In[98]:


new.shape


# In[99]:


full=pd.get_dummies(new,drop_first=False)
full.shape


# In[100]:


Train2 = full[full['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
Test2 = full[full['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()
print(Train2.shape)
print(Test2.shape)


# In[101]:


depVar = "Revenue.Grid"
train2X = Train2.drop([depVar], axis = 1).copy()
train2Y = Train2[depVar].copy()
test2X = Test2.drop([depVar], axis = 1).copy()
test2Y = Test2[depVar].copy()


# In[102]:


print(train2X.shape)
print(train2Y.shape)
print(test2X.shape)
print(test2Y.shape)


# In[103]:


RR=RandomForestClassifier(n_estimators=200)
RR.fit(train2X,train2Y)


# In[104]:


print("train",accuracy_score(train2Y,RR.predict(train2X)))
print("test",accuracy_score(test2Y,RR.predict(test2X)))
print(classification_report(train2Y, RR.predict(train2X)))
print(classification_report(test2Y,RR.predict(test2X)))
print("Confusion matrix:")
print(confusion_matrix(train2Y,RR.predict(train2X)))
print(confusion_matrix(test2Y,RR.predict(test2X)))



# Grid Search

param_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_leaf': [5, 10, 25, 50],
    'max_features': [5, 7, 9]}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=RRR,
    param_grid=param_grid,
    cv=3,  # Use 3-fold cross-validation
    scoring='accuracy',  # Use accuracy as the evaluation metric
    n_jobs=-1  # Use all available CPU cores
)

grid_search.fit(trainX, trainY)

grid_search.best_params_


RRRGRIDRESULTS= pd.DataFrame.from_dict(grid_search.cv_results_)
grid_search.cv_results_


RRR=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_features = 9 )
RRR.fit(trainX,trainY)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("train",accuracy_score(trainY,RRR.predict(trainX)))
print(confusion_matrix(trainY,RRR.predict(trainX)))
print(classification_report(trainY, RRR.predict(trainX)))
print("test",accuracy_score(testY,RRR.predict(testX)))
print(classification_report(testY,RRR.predict(testX)))
print("Confusion matrix:")
print(confusion_matrix(testY,RRR.predict(testX)))

Test_Pred = RRR.predict(testX)
Confusion_Mat = pd.crosstab(testY, Test_Pred) # R, C format (Actual = testY, Predicted = Test_Pred)
Confusion_Mat 


from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(RRR, feature_names = full.columns,class_names=['Disease', "No Disease"],filled=True);

trainY['Revenue.Grid'].replace({'0':'1', '1':'2'},inplace=True)

from statsmodels.api import Logit  
M1 = Logit(trainY, trainX) # (Dep_Var, Indep_Vars) # This is model definition
M1_Model = M1.fit() # This is model building
M1_Model.summary()
# In[105]:


from sklearn.ensemble import GradientBoostingClassifier
GD=GradientBoostingClassifier()
GD.fit(train2X,train2Y)


# In[106]:


from sklearn.model_selection import cross_val_score,cross_val_predict


# In[107]:


print("On_train",cross_val_score(GD, train2X, train2Y, cv=5, scoring='accuracy').mean())
# cross validation using cross_val_score
y_pred = cross_val_predict(GD, train2X, train2Y, cv=5)
print("Confusion matrix:")
print(confusion_matrix(train2Y, y_pred))

print("\nClassification report:")
print(classification_report(train2Y, y_pred))
print("******************************************")
# On test
from sklearn.model_selection import cross_val_score
print("Ontest",cross_val_score(GD, test2X, test2Y, cv=5, scoring='accuracy').mean())
y_pred1 = cross_val_predict(GD, test2X, test2Y, cv=5)
print("Confusion matrix:")
print(confusion_matrix(test2Y, y_pred1))

print("\nClassification report:")
print(classification_report(test2Y, y_pred1))

svc = SVC(C=30)
knc = KNeighborsClassifier(n_neighbors=7)
dtc = DecisionTreeClassifier(criterion='entropy',
 max_depth= 20,
 max_features= None,
 min_samples_leaf= 2,
 min_samples_split= 2,
 splitter= 'best')
lrc = LogisticRegression(C= 0.01, penalty = 'l2', solver = 'liblinear')
rfc = RandomForestClassifier(max_depth= 15,
 min_samples_leaf= 1,
 min_samples_split= 2,
 n_estimators= 200,random_state=123)
abc = AdaBoostClassifier(learning_rate= 1.0, n_estimators= 200)
bc = BaggingClassifier()
etc = ExtraTreesClassifier(n_estimators= 50,
    max_depth=  20,
    min_samples_split= 2,
    min_samples_leaf= 1,
    max_features= 'log2')
gbdt = GradientBoostingClassifier()
xgb = XGBClassifier()


# In[63]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BaggingClassifier': bc, 
    'ETC': etc,
    'GradientBoosting':gbdt
}


# In[64]:


def train_classifier(clf,trainX,trainY,testX,testY):
    clf.fit(trainX,trainY)
    y_pred = clf.predict(testX)
    accuracy = accuracy_score(testY,y_pred)
    precision = precision_score(testY,y_pred)
    recall=recall_score(testY,y_pred)
    f1=f1_score(testY,y_pred)
    
    
    return accuracy,precision,recall,f1


# In[65]:


accuracy_scores = []
precision_scores = []
recal_scores=[]
f1_scores=[]

for name,clf in clfs.items():
    
    current_accuracy,current_precision,current_recall,current_f1 = train_classifier(clf, trainX,trainY,testX,testY)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Recall - ",current_recall)
    print("f1 - ",current_f1)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recal_scores.append(current_recall)
    f1_scores.append(current_f1)


# In[68]:


performance_df=pd.DataFrame({'Algorithm':clfs.keys(),'f1_score':f1_scores,'Precision':precision_scores,'recall_score':recal_scores,'accuracy':accuracy_scores}).sort_values('f1_score',ascending=False)
performance_df1=pd.melt(performance_df, id_vars = "Algorithm")
performance_df


# In[67]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=9,aspect=1.3)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# # Voting

# In[120]:


from sklearn.ensemble import VotingClassifier
Voting=VotingClassifier(estimators=[('svc',svc),('kn',knc),('rf',rfc),('lg',lrc)],weights=([4,2,1,1]))
Voting.fit(trainX,trainY)


# In[121]:


print("train",accuracy_score(trainY,Voting.predict(trainX)))
print("Confusion matrix:")
print(confusion_matrix(trainY,Voting.predict(trainX)))
print(classification_report(trainY, Voting.predict(trainX)))
print("test",accuracy_score(testY,Voting.predict(testX)))
print(classification_report(testY,Voting.predict(testX)))
print(confusion_matrix(testY,Voting.predict(testX)))


# In[124]:


from mlxtend.plotting import plot_learning_curves

plot_learning_curves(trainX, trainY, testX, testY, Voting, print_model=False, style='ggplot')
plt.show()


# # Pipeline with stacking

# In[34]:


from sklearn.pipeline import Pipeline


# In[56]:


clf1 = etc = ExtraTreesClassifier(n_estimators= 50,
    max_depth=  20,
    min_samples_split= 2,
    min_samples_leaf= 1,
    max_features= 'log2',random_state=123)
clf6 = RandomForestClassifier(n_estimators=200,random_state=123)
clf2 = KNeighborsClassifier(n_neighbors=6)
clf1=SVC(C=10,random_state=123)
clf3=BaggingClassifier(random_state=123)
clf5=LogisticRegression(random_state=123,C=1)
clf4=DecisionTreeClassifier()


# In[57]:


step1=StackingClassifier(estimators=[('svc',clf1),('knn',clf2),('bagging',clf3),("dt",clf4),("lr",clf5)],final_estimator=clf6)


# In[45]:


pipe=Pipeline([
    ('stacking',step1)
])


# In[46]:


pipe.fit(trainX,trainY)


# In[47]:


print("train",accuracy_score(trainY,pipe.predict(trainX)))
print("Confusion matrix:")
print(confusion_matrix(trainY,pipe.predict(trainX)))
print(classification_report(trainY, pipe.predict(trainX)))
print("test",accuracy_score(testY,pipe.predict(testX)))
print(classification_report(testY,pipe.predict(testX)))
print(confusion_matrix(testY,pipe.predict(testX)))


# In[61]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
scores = []
for clf in [clf1, clf2, clf3,clf4,clf5, pipe]:
    clf.fit(trainX, trainY)
    if clf == step1:
        probas = clf.predict_proba(testX)
        score = roc_auc_score(testY, probas[:, 1])
    else:
        preds = clf.predict(testX)
        score = f1_score(testY, preds)
    scores.append(score)

# Create a bar chart showing the scores for each model
labels = ['SVC', 'KNN', 'BAGGING','DT','Logistic' ,'Stacked Model']
x_pos = np.arange(len(labels))
plt.bar(x_pos, scores)
plt.xticks(x_pos, labels,rotation=90)
plt.ylabel('Score')
plt.title('Performance of Base Models and Stacked Model')
plt.show()


# In[123]:


from mlxtend.plotting import plot_learning_curves

plot_learning_curves(trainX, trainY, testX, testY, pipe, print_model=False, style='ggplot')
plt.show()


# In[140]:


y_pred_prob=pipe.predict_proba(testX)[:,1]
df =pd.DataFrame({'Actual': testY, 'Predicted': y_pred_prob})
df['Decile'] = pd.qcut(df['Predicted'], 10, labels=False, duplicates='drop')
# Calculate the average predicted probability and the average actual outcome for each decile
decile_analysis = df.groupby('Decile').agg({'Actual': 'mean', 'Predicted': 'mean'}).reset_index()
# Plot the average predicted probability and the average actual outcome for each decile on a graph
decile_analysis.plot(x='Decile', y=['Actual', 'Predicted'], kind='bar')

# In[ ]:




