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


def load_masked_data(deceptive, truthful, split_by_sentence=False,padded=True,traintest_ratio=.8, truncate=True, max_length=500):
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
		tmp[-1]='[SEP]'
		dec.append(tmp)
		maxes.append(len(tmp))
		decmask.append(np.ones(len(tmp)))
	for para in trutext:
		tmp=tokenizer.tokenize(para)[:max_length]
		tmp[-1]='[SEP]'
		tru.append(tmp)
		maxes.append(len(tmp))
		trumask.append(np.ones(len(tmp)))

	decpos=[]
	trupos=[]
	posdec=[]
	postru=[]
	

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

	dec_test=dec[int(lendec*traintest_ratio):]
	tru_test=tru[int(lentru*traintest_ratio):]

	dec=dec[0:int(lendec*traintest_ratio)]
	tru=tru[0:int(lentru*traintest_ratio)]

	decmask_test=decmask[int(lendec*traintest_ratio):]
	trumask_test=trumask[int(lentru*traintest_ratio):]

	decmask=decmask[0:int(lendec*traintest_ratio)]
	trumask=trumask[0:int(lentru*traintest_ratio)]

	y_dec=np.ones(lendec)
	y_tru=np.zeros(lentru)

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

	

	x_train=x_train.long()
	x_test=x_test.long()

	y_train=torch.tensor(y_train).long()
	y_test=torch.tensor(y_test).long()

	x_train_mask=x_train_mask.long()
	x_test_mask=x_test_mask.long()

	dec=torch.tensor(dec).long()
	tru=torch.tensor(tru).long()

	decmask=torch.tensor(decmask).long()
	trumask=torch.tensor(trumask).long()

	dec_test=torch.tensor(dec_test).long()
	tru_test=torch.tensor(tru_test).long()

	decmask_test=torch.tensor(decmask_test).long()
	trumask_test=torch.tensor(trumask_test).long()
	return dec, tru, decmask, trumask, dec_test, tru_test, decmask_test, trumask_test, tokenizer