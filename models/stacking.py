import pickle
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split


def normalize(df):
	norms = []
	convert_dict = {}
	for col in df.columns:
		## REJECTING STRING DATATYPES
		if df[str(col)].dtype != 'object':
			mean = df[col].mean()
			max = df[col].max()
			min = df[col].min()
			convert_dict[str(col)] = float
			norms.append((mean, max, min))
		else:
			norms.append(-1)

	df = df.astype(convert_dict)

	for index, row in df.iterrows():
		for col in range(len(df.columns)):
			if norms[col] != -1:
				(mean, max, min) = norms[col]
				if (max - min) > 0:
					row[str(df.columns[col])] = (row[str(df.columns[col])] - mean)/(max-min)

	return df


## LOADING TRAINING DATA
train = pd.read_csv('/home/opvyas/kddcup.data.corrected',names =["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"])
train.head()
# normalize(train)
prots = pd.get_dummies(train['protocol_type'], drop_first=True)
serv = pd.get_dummies(train['service'], drop_first=True)
flg = pd.get_dummies(train['flag'], drop_first=True)
train = pd.concat([train, prots, serv, flg], axis=1)
train.drop(['protocol_type','service','flag'],axis=1,inplace=True)
train.head()
X = train.drop('label',axis=1)
y = train['label']

########### MLP ##############
filename = '../pickles/MLP_pickle.sav'
mlpmodel = pickle.load(open(filename, 'rb'))

########### NAIVE BAYES ##############
filename = '../pickles/NB_pickle.sav'
nbmodel = pickle.load(open(filename, 'rb'))

########## DECISION TREE ##############
filename = '../pickles/DT_pickle.sav'
dtmodel = pickle.load(open(filename, 'rb'))


######## RANDOM FOREST ################
filename = '../pickles/RF_pickle.sav'
rfmodel = pickle.load(open(filename, 'rb'))

training,valid,ytraining,yvalid = train_test_split(X,y,test_size=0.5)
preds_mlpmodel = mlpmodel.predict(valid)
preds_dtmodel = dtmodel.predict(valid)
preds_nbmodel = nbmodel.predict(valid)

## LOADING TEST DATA
test = pd.read_csv('/home/opvyas/corrected',names =["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"])
# normalize(test)
prots = pd.get_dummies(test['protocol_type'], drop_first=True)
serv = pd.get_dummies(test['service'])
gg = pd.DataFrame(pd.np.column_stack([serv, np.zeros((serv.shape[0], 4), dtype=int)]))
flg = pd.get_dummies(test['flag'], drop_first=True)
test = pd.concat([test, prots, gg, flg], axis=1)
test.drop(['protocol_type','service','flag'],axis=1,inplace=True)
XX = test.drop('label', axis=1)
yy = test['label']

## MAPPING LABELS TO NUMERICAL VALUES
ma = {}
rma = {}
i=0
for string in y:
	if string not in ma.keys():
		ma[string] = i
		rma[i] = string
		i+=1

test_preds_mlpmodel = mlpmodel.predict(XX)
test_preds_dtmodel = dtmodel.predict(XX)
test_preds_nbmodel = nbmodel.predict(XX)

for i in range(len(preds_mlpmodel)):
	preds_mlpmodel[i] = ma[preds_mlpmodel[i]]
for i in range(len(preds_dtmodel)):
	preds_dtmodel[i] = ma[preds_dtmodel[i]]
for i in range(len(preds_nbmodel)):
	preds_nbmodel[i] = ma[preds_nbmodel[i]]

for i in range(len(test_preds_mlpmodel)):
	test_preds_mlpmodel[i] = ma[test_preds_mlpmodel[i]]
for i in range(len(test_preds_dtmodel)):
	test_preds_dtmodel[i] = ma[test_preds_dtmodel[i]]
for i in range(len(test_preds_nbmodel)):
	test_preds_nbmodel[i] = ma[test_preds_nbmodel[i]]

stacked_predictions = np.column_stack((preds_mlpmodel,preds_dtmodel,preds_rfmodel))
test_stacked_predictions = np.column_stack((test_preds_mlpmodel,test_preds_dtmodel,test_preds_rfmodel))


meta_model = RandomForestClassifier()
meta_model.fit(stacked_predictions,yvalid)
result = meta_model.score(test_stacked_predictions,yy)
print("Accuracy using stacking - Random Forest last layer:" + str(result))

meta_model2 = tree.DecisionTreeClassifier()
meta_model2.fit(stacked_predictions,yvalid)
result = meta_model2.score(test_stacked_predictions,yy)
print("Accuracy using stacking - Decision Tree last layer:" + str(result))
