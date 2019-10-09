import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import matplotlib.pyplot as plt
import scipy.io as sio 

f = h5py.File('../RF_data_90.h5','r')
print(f.keys())

data = f['home'];
print(data.shape);
#print(np.mean(data[:,:,:,23]));
data = np.transpose(data,(3,2,1,0));
print(data.shape)
#print(np.mean(data[11,:,:,:]));
f.close()

'''
# multiply spectrum by f to whiten the spectrum
for i in range(55):
	data[:,:,:,i] = data[:,:,:,i]*(i+1);
'''

'''
# bin the spectrum into delta (1-4 Hz), theta (5-7 Hz), alpha (8-13 Hz), beta (14-30 Hz) and gamma (31-55 Hz)
data_freq_binned = np.zeros((data.shape[0],data.shape[1],data.shape[2],5));
print(data_freq_binned.shape)
data_freq_binned[:,:,:,0] = np.sum(data[:,:,:,0:3],3)
data_freq_binned[:,:,:,1] = np.sum(data[:,:,:,4:6],3)
data_freq_binned[:,:,:,2] = np.sum(data[:,:,:,7:12],3)
data_freq_binned[:,:,:,3] = np.sum(data[:,:,:,13:29],3)
data_freq_binned[:,:,:,4] = np.sum(data[:,:,:,30:54],3)
data = np.reshape(data_freq_binned,(25*7500,64,5),order='C');
data = np.reshape(data,(25*7500,64*5),order='C');
'''

##########arna = raw_input()########################### proxy for breakpoint
data = np.reshape(data,(25*7500,64,55),order='C');
data = np.reshape(data,(25*7500,64*55),order='C');
print(data.shape)
#print(np.mean(data[7500*10:7500*11,:]))
labels = np.concatenate((np.zeros((13*7500,1)),np.ones((12*7500,1))))
labels = np.ravel(labels)
print(labels.shape)

# train_feats,test_feats,train_labels,test_labels = train_test_split(data,labels, test_size = 0.2, random_state = 42)

folds = 10;
test_con_subj_list = np.array([12, 2, 8, 3, 1,12, 1, 9,12, 4]); test_con_subj_list -= 1;	# list of CON subjects in test set for 10 folds (used in Torch for random seed 13)
test_exe_subj_list = np.array([25,15,19,14,22,14,20,17,18,15]); test_exe_subj_list -= 1;	# list of EXE subjects in test set for 10 folds (used in Torch for random seed 13)
train_acc_arr = [];
test_acc_arr = [];
for f in range(folds):
	
	# splitting train-test for cross-validation -> comment out this section (before printing train_feats.shape) if doing 80-20 split
	test_con_subj = test_con_subj_list[f]	# random.randint(0,12);
	test_exe_subj = test_exe_subj_list[f]	# random.randint(13,24);
	print "Fold ",f," starting: Test subjects -> ",test_con_subj," and ",test_exe_subj;
	train_con_feats = np.concatenate((data[0:test_con_subj*7500,:],data[(test_con_subj+1)*7500:13*7500,:]))	
	train_exe_feats = np.concatenate((data[13*7500:test_exe_subj*7500,:],data[(test_exe_subj+1)*7500:25*7500,:]))	
	train_con_labels = np.concatenate((labels[0:test_con_subj*7500],labels[(test_con_subj+1)*7500:13*7500]))	
	train_exe_labels = np.concatenate((labels[13*7500:test_exe_subj*7500],labels[(test_exe_subj+1)*7500:25*7500]))	
	test_con_feats = data[test_con_subj*7500:(test_con_subj+1)*7500,:]
	test_exe_feats = data[test_exe_subj*7500:(test_exe_subj+1)*7500,:]
	test_con_labels = labels[test_con_subj*7500:(test_con_subj+1)*7500]
	test_exe_labels = labels[test_exe_subj*7500:(test_exe_subj+1)*7500]
	print(np.shape(train_con_feats),np.shape(train_exe_feats),np.shape(test_con_feats),np.shape(test_exe_feats))
	
	train_feats = np.concatenate((train_con_feats,train_exe_feats));
	train_labels = np.concatenate((train_con_labels,train_exe_labels));
	test_feats = np.concatenate((test_con_feats,test_exe_feats));
	test_labels = np.concatenate((test_con_labels,test_exe_labels));
	
	print('Training Features Shape:', train_feats.shape)
	print('Training Labels Shape:', train_labels.shape)
	print('Testing Features Shape:', test_feats.shape)
	print('Testing Labels Shape:', test_labels.shape)

	rf = RandomForestClassifier(n_estimators = 100, max_depth = 15, random_state = 42)
	rf.fit(train_feats,train_labels)

	predictions = rf.predict(test_feats)
	accuracy = 1-np.sum(np.abs(predictions-test_labels))/np.shape(test_labels)[0]
	print("Testing accuracy",accuracy)
	# avg_test_acc = avg_test_acc + accuracy;
	test_acc_arr.append(accuracy)

	predictions = rf.predict(train_feats)
	accuracy = 1-np.sum(np.abs(predictions-train_labels))/np.shape(train_labels)[0]
	print("Training accuracy",accuracy)
	# avg_train_acc = avg_train_acc + accuracy;
	train_acc_arr.append(accuracy)
	
	'''
	importances = rf.feature_importances_
	print(np.shape(importances))
	#print(importances)
	#importances = np.reshape(importances,(64,55)) # use when freq not binned
	importances = np.reshape(importances,(64,5)) # use when freq binned
	print(np.shape(importances))
	sio.savemat('/home/arna/Desktop/RF_imp.mat',{'imp':importances});
	importance_freq = np.mean(importances,axis=0)
	print(np.shape(importance_freq))
	plt.show()
	importance_elec = np.mean(importances,axis=1)
	print(np.shape(importance_elec))
	plt.figure(1)
	plt.subplot(211)
	plt.plot(importance_freq)
	plt.xlabel('Frequency')
	plt.subplot(212)
	plt.plot(importance_elec)
	plt.xlabel('Electrodes')
	plt.show()
	'''
	
train_acc_arr = np.array(train_acc_arr)
test_acc_arr = np.array(test_acc_arr)
avg_train_acc = np.mean(train_acc_arr); std_train_acc = np.std(train_acc_arr)
avg_test_acc = np.mean(test_acc_arr); std_test_acc = np.std(test_acc_arr)
print "Average Training Accuracy: ", avg_train_acc, ", Stdev Training Accuracy: ", std_train_acc
print "Average Testing Accuracy: ", avg_test_acc, ", Stdev Testing Accuracy: ", std_test_acc