import torch
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from keras.preprocessing import sequence
from torch.autograd import Variable
import argparse

from tensorboardX import SummaryWriter
import datetime,socket,os

def mask_pos(pair, pos):
	tmp=pair[0]
	if pair[1] == pos:
		tmp='[MASK]'
	return tmp

#takes two pickled files of truth and deception and turns them into a usable dataset
def load_masked_data(deceptive, truthful, padded=True,traintest_ratio=.8, truncate=True, max_length=500,masked_pos=None):
	with open(deceptive,'rb') as f:
		dectext=pickle.load(f)
	with open(truthful,'rb') as f:
		trutext=pickle.load(f)
	tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

	
	

	dec=[]
	tru=[]
	
	decmask=[]
	trumask=[]

	
	maxes=[]
	for para in dectext:

		tmp=tokenizer.tokenize(para)[:max_length]
		dec.append(tmp)
		maxes.append(len(tmp))
		decmask.append(np.ones(len(tmp)))
	for para in trutext:
		tmp=tokenizer.tokenize(para)[:max_length]
		tru.append(tmp)
		maxes.append(len(tmp))
		trumask.append(np.ones(len(tmp)))

	decpos=[]
	trupos=[]
	posdec=[]
	postru=[]
	


	if masked_pos!=None:
		import nltk
		for para in dec:
			decpos.append(nltk.pos_tag(para))
		for para in tru:
			trupos.append(nltk.pos_tag(para))
		#mask given part of speech
		for i in range(len(dec)):
			dec[i]=[mask_pos(decpos[i][j],masked_pos) if dec[i][j] not in ['[CLS]','[SEP]'] else dec[i][j] for j in range(len(decpos[i]))]
			tru[i]=[mask_pos(trupos[i][j],masked_pos) if tru[i][j] not in ['[CLS]','[SEP]'] else tru[i][j] for j in range(len(trupos[i]))]
	for i in range(len(dec)):
		dec[i]=tokenizer.convert_tokens_to_ids(dec[i])
	for i in range(len(tru)):
		tru[i]=tokenizer.convert_tokens_to_ids(tru[i])
	maxlen=np.max(maxes)

	if truncate == True and maxlen > max_length:
		maxlen=max_length

	dec=sequence.pad_sequences(dec,maxlen, padding='post', truncating='post')
	tru=sequence.pad_sequences(tru,maxlen, padding='post', truncating='post')

	decmask=sequence.pad_sequences(decmask,maxlen, padding='post', truncating='post')
	trumask=sequence.pad_sequences(trumask,maxlen, padding='post', truncating='post')

	

	lendec=len(dec)
	lentru=len(tru)

	y_dec=np.ones(lendec)
	y_tru=np.zeros(lentru)


	#merge dec and tru into full datasets
	x_train=np.concatenate((dec[0:int(lendec*traintest_ratio)],tru[0:int(lentru*traintest_ratio)]))
	x_test=np.concatenate((dec[int(lendec*traintest_ratio):],tru[int(lentru*traintest_ratio):]))

	x_train_mask=np.concatenate((decmask[0:int(lendec*traintest_ratio)],trumask[0:int(lentru*traintest_ratio)]))
	x_test_mask=np.concatenate((decmask[int(lendec*traintest_ratio):],trumask[int(lentru*traintest_ratio):]))

	y_train=np.concatenate((y_dec[0:int(lendec*traintest_ratio)],y_tru[0:int(lentru*traintest_ratio)]))
	y_test=np.concatenate((y_dec[int(lendec*traintest_ratio):],y_tru[int(lentru*traintest_ratio):]))

	# print(len(x_train))
	# print(len(x_train_mask))
	# print(len(x_test))
	# print(len(x_test_mask))
	for i in range(len(x_train)):
		x_train[i]=torch.tensor(x_train[i])

	for i in range(len(x_test)):
		x_test[i]=torch.tensor(x_test[i])

	x_train=torch.tensor(x_train)
	x_test=torch.tensor(x_test)

	x_train_mask=torch.tensor(x_train_mask)
	x_test_mask=torch.tensor(x_test_mask)

	
#convert to integers
	x_train=x_train.long()
	x_test=x_test.long()

	y_train=torch.tensor(y_train).long()
	y_test=torch.tensor(y_test).long()

	x_train_mask=x_train_mask.long()
	x_test_mask=x_test_mask.long()

	return x_train, x_test, x_train_mask, x_test_mask, y_train, y_test, tokenizer