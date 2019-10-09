import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import matplotlib.pyplot as plt
import scipy.io as sio 
from mne.decoding import CSP
import mne
from tqdm import tqdm
mne.set_log_level('WARNING')

f = h5py.File('../RF_data_90.h5','r')
print(f.keys())

data = f['home'];
print(data.shape);
#print(np.mean(data[:,:,:,23]));
data = np.transpose(data,(3,2,1,0));
print(data.shape)
#print(np.mean(data[11,:,:,:]));
f.close()


# multiply spectrum by f to whiten the spectrum
for i in range(55):
	data[:,:,:,i] = data[:,:,:,i]*(i+1);


data = np.reshape(data,(25*7500,64,55),order='C');
print(data.shape)

labels = np.concatenate((np.zeros((13*7500,1)),np.ones((12*7500,1))))
labels = np.ravel(labels)

folds = 10
test_con_subj_list = np.array([12, 2, 8, 3, 1,12, 1, 9,12, 4]); test_con_subj_list -= 1;	# list of CON subjects in test set for 10 folds (used in Torch for random seed 13)
test_exe_subj_list = np.array([25,15,19,14,22,14,20,17,18,15]); test_exe_subj_list -= 1;	# list of EXE subjects in test set for 10 folds (used in Torch for random seed 13)
train_acc_arr = [];
test_acc_arr = [];

#train_feats,test_feats,train_labels,test_labels = train_test_split(data,labels, test_size = 0.2, random_state = 42)
#print(train_feats.shape)

for f in tqdm(range(folds)):
	#'''
	# splitting train-test for cross-validation -> comment out this section (before printing train_feats.shape) if doing 80-20 split
	test_con_subj = test_con_subj_list[f]	# random.randint(0,12);
	test_exe_subj = test_exe_subj_list[f]	# random.randint(13,24);
	tqdm.write("Fold "+str(f)+" starting: Test subjects -> "+str(test_con_subj)+" and "+str(test_exe_subj));
	train_con_feats = np.concatenate((data[0:test_con_subj*7500,:],data[(test_con_subj+1)*7500:13*7500,:]))	
	train_exe_feats = np.concatenate((data[13*7500:test_exe_subj*7500,:],data[(test_exe_subj+1)*7500:25*7500,:]))	
	train_con_labels = np.concatenate((labels[0:test_con_subj*7500],labels[(test_con_subj+1)*7500:13*7500]))	
	train_exe_labels = np.concatenate((labels[13*7500:test_exe_subj*7500],labels[(test_exe_subj+1)*7500:25*7500]))	
	test_con_feats = data[test_con_subj*7500:(test_con_subj+1)*7500,:]
	test_exe_feats = data[test_exe_subj*7500:(test_exe_subj+1)*7500,:]
	test_con_labels = labels[test_con_subj*7500:(test_con_subj+1)*7500]
	test_exe_labels = labels[test_exe_subj*7500:(test_exe_subj+1)*7500]
	tqdm.write(str(np.shape(train_con_feats))+" "+str(np.shape(train_exe_feats))+" "+str(np.shape(test_con_feats))+" "+str(np.shape(test_exe_feats)))
	
	train_feats = np.concatenate((train_con_feats,train_exe_feats));
	train_labels = np.concatenate((train_con_labels,train_exe_labels));
	test_feats = np.concatenate((test_con_feats,test_exe_feats));
	test_labels = np.concatenate((test_con_labels,test_exe_labels));
	#'''
	tqdm.write('Training Features Shape:'+ str(train_feats.shape))
	tqdm.write('Training Labels Shape:'+ str(train_labels.shape))
	tqdm.write('Testing Features Shape:'+ str(test_feats.shape))
	tqdm.write('Testing Labels Shape:'+ str(test_labels.shape))

	csp = CSP(n_components=50, norm_trace=False)
	rf = RandomForestClassifier(n_estimators = 50, max_depth = 7, random_state = 42)

	X_train = csp.fit_transform(train_feats,train_labels)
	X_test = csp.transform(test_feats)
	rf.fit(X_train,train_labels)

	predictions = rf.predict(X_test)
	accuracy = 1-np.sum(np.abs(predictions-test_labels))/np.shape(test_labels)[0]
	tqdm.write("Testing accuracy "+str(accuracy))
	# avg_test_acc = avg_test_acc + accuracy;
	test_acc_arr.append(accuracy)

	predictions = rf.predict(X_train)
	accuracy = 1-np.sum(np.abs(predictions-train_labels))/np.shape(train_labels)[0]
	tqdm.write("Training accuracy "+str(accuracy))
	# avg_train_acc = avg_train_acc + accuracy;
	train_acc_arr.append(accuracy)
	
train_acc_arr = np.array(train_acc_arr)
test_acc_arr = np.array(test_acc_arr)
avg_train_acc = np.mean(train_acc_arr); std_train_acc = np.std(train_acc_arr)
avg_test_acc = np.mean(test_acc_arr); std_test_acc = np.std(test_acc_arr)
print "Average Training Accuracy: ", avg_train_acc, ", Stdev Training Accuracy: ", std_train_acc
print "Average Testing Accuracy: ", avg_test_acc, ", Stdev Testing Accuracy: ", std_test_acc