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


def load_excised_data(deceptive, truthful, excise_dec, excise_tru,padded=True,traintest_ratio=.8, truncate=True, max_length=500,masked_pos=None):
	with open(deceptive,'rb') as f:
		dectext=pickle.load(f)
	with open(truthful,'rb') as f:
		trutext=pickle.load(f)
	tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

	with open(excise_dec,'rb') as f:
		ex_dec_text=pickle.load(f)
	with open(excise_tru,'rb') as f:
		ex_tru_text=pickle.load(f)

	dec=[]
	tru=[]
	
	ex_dec=[]
	ex_tru=[]

	decmask=[]
	trumask=[]

	ex_decmask=[]
	ex_trumask=[]

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
	#excised
	for para in ex_dec_text:
		tmp=para[:max_length]
		ex_dec.append(tmp)
		maxes.append(len(tmp))
		ex_decmask.append(np.ones(len(tmp)))
	for para in ex_tru_text:
		tmp=para[:max_length]
		ex_tru.append(tmp)
		maxes.append(len(tmp))
		ex_trumask.append(np.ones(len(tmp)))


	for i in range(len(dec)):
		# print(dec[i])
		dec[i]=tokenizer.convert_tokens_to_ids(dec[i])
	for i in range(len(tru)):
		tru[i]=tokenizer.convert_tokens_to_ids(tru[i])
	for i in range(len(ex_dec)):
		ex_dec[i]=tokenizer.convert_tokens_to_ids(ex_dec[i])
	for i in range(len(ex_tru)):
		ex_tru[i]=tokenizer.convert_tokens_to_ids(ex_tru[i])
	maxlen=np.max(maxes)

	if truncate == True and maxlen > max_length:
		maxlen=max_length

	dec=sequence.pad_sequences(dec,maxlen, padding='post', truncating='post')
	tru=sequence.pad_sequences(tru,maxlen, padding='post', truncating='post')

	decmask=sequence.pad_sequences(decmask,maxlen, padding='post', truncating='post')
	trumask=sequence.pad_sequences(trumask,maxlen, padding='post', truncating='post')

	print(len(ex_dec))

	ex_dec=sequence.pad_sequences(ex_dec,maxlen, padding='post', truncating='post')
	ex_tru=sequence.pad_sequences(ex_tru,maxlen, padding='post', truncating='post')

	ex_decmask=sequence.pad_sequences(ex_decmask,maxlen, padding='post', truncating='post')
	ex_trumask=sequence.pad_sequences(ex_trumask,maxlen, padding='post', truncating='post')

	lendec=len(dec)
	lentru=len(tru)

	y_dec=np.ones(lendec)
	y_tru=np.zeros(lentru)

	y_ex_dec=np.ones(len(ex_dec))
	y_ex_tru=np.zeros(len(ex_tru))

	combined=np.concatenate((dec,tru))
	combined_mask=np.concatenate((decmask,trumask))

	ex_combined=np.concatenate((ex_dec,ex_tru))
	ex_combined_mask=np.concatenate((ex_decmask,ex_trumask))
	
	y_comb=np.concatenate((y_dec,y_tru))
	y_ex_comb=np.concatenate((np.ones(len(ex_dec)),np.zeros(len(ex_tru))))

	# print(len(x_train))
	# print(len(x_train_mask))
	# print(len(x_test))
	# print(len(x_test_mask))
	for i in range(len(dec)):
		dec[i]=torch.tensor(dec[i])

	for i in range(len(tru)):
		tru[i]=torch.tensor(tru[i])

	for i in range(len(ex_dec)):
		ex_dec[i]=torch.tensor(ex_dec[i])

	for i in range(len(ex_tru)):
		ex_tru[i]=torch.tensor(ex_tru[i])

	dec=torch.tensor(dec).long()
	ex_dec=torch.tensor(ex_dec).long()
	tru=torch.tensor(tru).long()
	ex_tru=torch.tensor(ex_tru).long()

	decmask=torch.tensor(decmask).long()
	trumask=torch.tensor(trumask).long()
	ex_decmask=torch.tensor(ex_decmask).long()
	ex_trumask=torch.tensor(ex_trumask).long()

	return  dec, decmask, tru, trumask, y_dec, y_tru, ex_dec, ex_tru, ex_decmask, ex_trumask, y_ex_dec, y_ex_tru