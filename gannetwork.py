import torch
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from modelingref import *
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
#from keras.preprocessing import sequence
from torch.autograd import Variable
import argparse

from tensorboardX import SummaryWriter
import datetime,socket,os
from torch.utils.data import DataLoader
from test_bert_generate import *


def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x





t_total=800



from torch.nn import Parameter
from torch import FloatTensor

def new_parameter(*size):
    out = Parameter(FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out



class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)

        return condensed_x



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.bert = BertModelFloater.from_pretrained('bert-base-uncased') #changed to Floater

		self.lstm=nn.LSTM(input_size=768, hidden_size=384, num_layers=2, dropout=.5, bidirectional=True)
		self.attention=Attention(768)
		self.classifier=nn.Linear(768,2)

	def forward(self, input, attention_mask=None):

		x=input
		if input.dtype==torch.int64:
			x=self.bert.embeddings(x)

		_,x=self.bert(x, attention_mask=attention_mask)

		x,(h,c)=self.lstm(x.unsqueeze(0))


		x=self.attention(x.view(x.shape[1],1,768))

		x=self.classifier(x)

		return x


		
from torch.utils.data import Dataset
class MDataset(Dataset):
	def __init__(self, train_data, y_data, train_mask):
		self.data=train_data
		self.labels=y_data
		self.mask=train_mask
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample={'data': self.data[idx],'label':self.labels[idx],'mask':self.mask[idx]}
		return sample

class TestDataset(Dataset):
	def __init__(self, train_data, y_data, train_mask, index):
		self.data=train_data
		self.labels=y_data
		self.mask=train_mask
		self.index=index
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		sample={'data': self.data[idx],'label':self.labels[idx],'mask':self.mask[idx], 'index':self.index[idx]}
		return sample



discriminator_cuda=0
generator_cuda=1


#raise Exception
discriminator=Net().cuda(discriminator_cuda)
discriminator.load_state_dict(torch.load('trained_discriminator.pt'))
loss_function=nn.CrossEntropyLoss().cuda(discriminator_cuda)
optimizer=optim.Adam(discriminator.parameters(), lr=.5e-5, weight_decay=1e-6)
#optimizer=optim.Adagrad(model.parameters(), lr=.5)



global_step=0
bs=40
rev_length=50


from excise_load import *
from load_gan_data import *

dec, tru, decmask, trumask, dec_test, tru_test, decmask_test, trumask_test, tokenizer=load_masked_data('berted_deception.pkl','berted_truthful.pkl', max_length=rev_length)

# dec, decmask, tru, trumask, y_dec, y_tru, ex_dec, ex_tru, ex_decmask, ex_trumask, y_ex_dec, y_ex_tru=load_excised_data(
# 	'berted_deception.pkl','berted_truthful.pkl','berted_tokenized_masked_deceptive_sentences.pkl','berted_tokenized_masked_truthful_sentences.pkl', max_length=rev_length, masked_pos=None)



from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()

with open('berted_deception.pkl','rb') as f:
	decrevs=pickle.load(f)

with open('berted_truthful.pkl','rb') as f:
	trurevs=pickle.load(f)

all_reviews=decrevs+trurevs

vectorizer.fit_transform(all_reviews)

vocab=[x for x in vectorizer.vocabulary_]

vocab_len=len(vocab)


#pick which type you are generating
trudec='tru'

if trudec=='dec':
	training_dataset=MDataset(dec.cuda(discriminator_cuda),torch.zeros(len(dec)).long().cuda(discriminator_cuda), decmask.cuda(discriminator_cuda))
else:
	training_dataset=MDataset(tru.cuda(discriminator_cuda),torch.zeros(len(tru)).long().cuda(discriminator_cuda), trumask.cuda(discriminator_cuda))
trainloader=DataLoader(training_dataset,batch_size=bs,shuffle=True,drop_last=False)


model_version = 'bert-base-uncased'
generator=BertForMaskedLMOptions.from_pretrained(model_version).cuda(generator_cuda)
gen_optim=optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-6)
gen_loss=nn.CrossEntropyLoss().cuda(generator_cuda)

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

seed_text = "[CLS]".split()
# gen_max_len = 48
seed_text = "[CLS] "+str(tokenizer.convert_ids_to_tokens([np.random.randint(1996,29611)])[0])
seed_text=seed_text.split()
gen_max_len = 49-len(seed_text)
temp=1.0
# n_samples = len(dec)
n_samples=40
# gen_batch_size = 100
gen_batch_size = 40

top_k = 100
temperature = 0.7

leed_out_len = 5 # max_len
burnin = 250
sample = True
max_iter = 500

clstok=torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]'))).long().cuda()

# raise Exception

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')



comment='_'+str(trudec)
log_dir = os.path.join('runs/'+ 'bert_' + current_time + '_' + socket.gethostname() + comment)
print(log_dir)
writer = SummaryWriter(log_dir = log_dir)

import sys


paramtest=[]

testep=0
for epoch in range(10+1):
	#make random seed
	seed_text = "[CLS] "+str(tokenizer.convert_ids_to_tokens([np.random.randint(1996,29611)])[0])
	seed_text=seed_text.split()
	paramtest.append(generator.parameters())
	discriminator.train()
	generator.train()
	print('Epoch: '+str(epoch+1))
	val_loss=0
	tr_loss = 0
	#-----------
	#in case of untrained discriminator
	#-----------
	# for batch in trainloader:
	# 	print('-', end='')
	# 	sys.stdout.flush()

	# 	discriminator.zero_grad()
	# 	optimizer.zero_grad()
			
			
	# 	#print(tar)
	# 	y_prd=discriminator(batch['data'],attention_mask=batch['mask'])

		

	# 	loss=loss_function(y_prd, batch['label'])

	# 	loss.backward()
	# 	tr_loss += loss.item()


	# 	optimizer.step()

	# # seed_seq=torch.LongTensor(800,rev_length-1).random_(0,vocab_len).cuda()

	# # gen_seq=[generator(x.unsqueeze(0)).detach() for x in seed_seq]
	# # gen_seq=torch.stack(gen_seq).squeeze(1)
	# # gen_seq=generator(seed_seq)
	# # word_choices=torch.argmax(gen_seq,2)

	# # cls_tensor=torch.ones(800).long().cuda()*clstok

	# # gen_all=torch.cat((cls_tensor.unsqueeze(1),word_choices),dim=1)
	# # gen_revs=gen_all[0:int(.8*800)]
	# # gen_test=gen_all[int(.8*800):]
	# gen_revs, raw_out=generate(model=generator, n_samples=n_samples, seed_text=seed_text, batch_size=gen_batch_size, max_len=gen_max_len,
 #                          sample=sample, top_k=top_k, temperature=temp, burnin=burnin, max_iter=max_iter,
 #                          cuda=True)
	# # for i in range(15):
	# # 	ngen_revs, nraw_out=generate(model=generator, n_samples=n_samples, seed_text=seed_text, batch_size=gen_batch_size, max_len=gen_max_len,
 # #                          sample=sample, top_k=top_k, temperature=temp, burnin=burnin, max_iter=max_iter,
 # #                          cuda=True)
	# # 	gen_revs=gen_revs+ngen_revs
	# # 	raw_out=torch.cat((raw_out,nraw_out))
	# gen_revs=tokenize_batch(gen_revs)
	# gen_revs=torch.tensor(gen_revs).long()
	# noise_set=MDataset(raw_out.cuda(),torch.ones(len(raw_out)).long().cuda(), torch.ones(len(raw_out),rev_length).long().cuda())
	# noiseloader=DataLoader(noise_set,batch_size=bs,shuffle=True,drop_last=False)

	# for batch in noiseloader:
	# 	print('-', end='')
	# 	sys.stdout.flush()
	# 	discriminator.zero_grad()
	# 	optimizer.zero_grad()
			
			
	# 	#print(tar)
	# 	y_prd=discriminator(batch['data'],batch['mask'])

		

	# 	loss=loss_function(y_prd, batch['label'])

	# 	loss.backward()
	# 	tr_loss += loss.item()


	# 	optimizer.step()

	#generate reviews
	gen_revs, raw_out=generate(model=generator, n_samples=n_samples, seed_text=seed_text, batch_size=gen_batch_size, max_len=gen_max_len,
                          sample=sample, top_k=top_k, temperature=temp, burnin=burnin, max_iter=max_iter,
                          cuda=True)

	gen_revs=tokenize_batch(gen_revs)
	gen_revs=torch.tensor(gen_revs).long()

	noise_set=MDataset(raw_out.cuda(),torch.zeros(len(raw_out)).long().cuda(), torch.ones(800,rev_length).long().cuda())
	noiseloader=DataLoader(noise_set,batch_size=bs,shuffle=True,drop_last=False)


	print('\nTraining Generator\n')
	generator.train()
	for batch in noiseloader:
		
		print('-', end='')
		sys.stdout.flush()
		discriminator.zero_grad()
		# print(list(generator.parameters())[0].grad)
		generator.zero_grad()
		gen_optim.zero_grad()
			
			
		#print(tar)
		y_prd=discriminator(batch['data'], batch['mask'])

		

		loss=gen_loss(y_prd, batch['label'])

		loss.backward()
		tr_loss += loss.item()


		gen_optim.step()
	print('\n')

	gen_revs, raw_out=generate(model=generator, n_samples=n_samples, seed_text=seed_text, batch_size=gen_batch_size, max_len=gen_max_len,
                          sample=sample, top_k=top_k, temperature=temp, burnin=burnin, max_iter=max_iter,
                          cuda=True)

	gen_revs=tokenize_batch(gen_revs)
	gen_revs=torch.tensor(gen_revs).long()

	testinds=torch.randint(0,len(dec_test),(40,))
	dec_test_valid=dec_test[testinds]
	# test_set=TestDataset(torch.cat((dec_test_valid.cuda(),gen_revs.cuda())),torch.cat((torch.zeros(len(dec_test_valid)).long().cuda(),torch.ones(len(gen_revs)).long().cuda())), torch.ones(len(dec_test_valid)+len(gen_revs),rev_length).long().cuda(),
	# 	np.arange(len(dec_test_valid)+len(gen_revs)))
	test_set_real=TestDataset(dec_test_valid.cuda(),torch.zeros(len(dec_test_valid)).long().cuda(),torch.ones(len(dec_test_valid), rev_length).long().cuda(),np.arange(len(dec_test_valid)))
	test_set_gen=TestDataset(raw_out.cuda().detach(),torch.ones(len(raw_out)).long().cuda(),torch.ones(len(raw_out), rev_length).long().cuda(),np.arange(len(raw_out)))
	testloader_real=DataLoader(test_set_real,batch_size=bs,shuffle=True,drop_last=False)
	testloader_gen=DataLoader(test_set_gen,batch_size=bs,shuffle=True,drop_last=False)

	if epoch%1==0:
		discriminator.eval()
		total_samples=0
		accnum=0
		print('\nTesting\n')
		foolind=[]
		for batch in testloader_real:
			y_prd=discriminator(batch['data'], batch['mask'])

			corrects=torch.argmax(y_prd,1)==batch['label']
			accnum+=torch.sum(corrects).item()
			total_samples+=bs

		for batch in testloader_gen:
			y_prd=discriminator(batch['data'], batch['mask'])

			corrects=torch.argmax(y_prd,1)==batch['label']
			accnum+=torch.sum(corrects).item()
			total_samples+=bs
			for n in range(len(batch['label'])):
				if batch['label'][n]==1 and torch.argmax(y_prd,1)[n]==0:
					# foolind=batch['index'][n]-len(dec_test)
					foolind.append(batch['index'][n]-len(dec_test_valid))
		print('Accuracy: '+str(accnum/total_samples))
		# print(foolind)
		last_rev=[]
		if len(foolind)>0:
			for fooler in foolind:
				last_rev.append(tokenizer.convert_ids_to_tokens(gen_revs[fooler].cpu().numpy()))
		print(last_rev)
		writer.add_scalar('bert_/GAN/accuracy',accnum/total_samples, epoch)
		# model.train()
		del gen_revs
		del raw_out
		del dec_test_valid
		torch.cuda.empty_cache()
with open(trudec+'_gan_last_revs.pkl','wb') as f:
	pickle.dump(last_rev,f)
