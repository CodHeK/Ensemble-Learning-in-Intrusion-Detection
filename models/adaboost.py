from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

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
X = train.drop('label', axis=1)
y = train['label']


## LOADING TEST DATA
test = pd.read_csv('/home/opvyas/corrected',names =["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"])
# normalize(test)
prots = pd.get_dummies(test['protocol_type'], drop_first=True)
serv = pd.get_dummies(test['service'])
gg = pd.DataFrame(pd.np.column_stack([serv, np.zeros((serv.shape[0], 4), dtype=int)]))
flg = pd.get_dummies(test['flag'], drop_first=True)
test = pd.concat([test, prots, gg, flg], axis=1)
test.drop(['protocol_type','service','flag'],axis=1,inplace=True)
print(prots.shape, serv.shape, flg.shape)
XX = test.drop('label', axis=1)
yy = test['label']

clf_dt = DecisionTreeClassifier(max_depth=3)
bclf = AdaBoostClassifier(base_estimator=clf_dt,n_estimators=50)
bclf.fit(X,y)
print("Decision tree using AdaBoost:" + str(bclf.score(XX,yy)))
