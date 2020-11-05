import torch
import pickle
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
#from keras.preprocessing import sequence
from torch.autograd import Variable
import argparse

from tensorboardX import SummaryWriter
import datetime,socket,os
from torch.utils.data import DataLoader



# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')






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
		self.bert = BertModel.from_pretrained('bert-base-uncased')


		self.lstm=nn.LSTM(input_size=768, hidden_size=384, num_layers=2, dropout=.5, bidirectional=True)
		self.attention=Attention(768)
		self.classifier=nn.Linear(768,2)


	def forward(self, input, attention_mask):
		_,x=self.bert(input, attention_mask=attention_mask)


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


#intitialize network
model=Net().cuda()
loss_function=nn.CrossEntropyLoss().cuda()
optimizer=optim.Adam(model.parameters(), lr=.5e-5, weight_decay=1e-6)
#list of available parts of speech
pos_list=[None,
'CC',
'CD',
'DT',
'EX',
'FW',
'IN',
'JJ',
'JJR',
'JJS',
'LS',
'MD',
'NN',
'NNS',
'NNP',
'NNPS',
'PDT',
'POS',
'PRP',
'PRP$',
'RB',
'RBR',
'RBS',
'RP',
'TO',
'UH',
'VB',
'VBD',
'VBG',
'VBN',
'VBP',
'VBZ',
'WDT',
'WP',
'WP$',
'WRB']

global_step=0
bs=5


import sys


#load data from pickled files
from load_bert_data import *


x_train, x_test, x_train_mask, x_test_mask, y_train, y_test, tokenizer=load_masked_data('berted_deception.pkl','berted_truthful.pkl', max_length=300, masked_pos=None)#subsitute None for a given part of speech in the list

training_dataset=MDataset(x_train.cuda(),y_train.cuda(), x_train_mask.cuda())
trainloader=DataLoader(training_dataset,batch_size=bs,shuffle=True,drop_last=False)

#tensorboard settings
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
comment='_'+str(None)
log_dir = os.path.join('runs/'+ 'bert_' + current_time + '_' + socket.gethostname() + comment)
print(log_dir)
writer = SummaryWriter(log_dir = log_dir)

testep=0

#train for 100 epochs
for epoch in range(101):
	model.train()

	print('Epoch: '+str(epoch+1))
	val_loss=0
	tr_loss = 0
	for batch in trainloader:
		print('-', end='')
		sys.stdout.flush()

		model.zero_grad()
		optimizer.zero_grad()
			
			
		#pass batch data and mask to network
		y_prd=model(batch['data'],attention_mask=batch['mask'])

		

		loss=loss_function(y_prd, batch['label'])

		loss.backward()
		tr_loss += loss.item()


		optimizer.step()

	print('\n')
	#test every 5 epochs
	if epoch%5==0:
		print('testing time')
		model.eval()
		res=[]
		for i in range(len(x_test)):
			y_prd=model(x_test[i].unsqueeze(0).cuda(), x_test_mask[i].unsqueeze(0).cuda())
			_, indmax=torch.max(y_prd[0],0)
			res.append(indmax.item())
		res=np.array(res)
		res=torch.tensor(res)

		corrects=res==y_test.long()

		acc=torch.sum(corrects).numpy()/float(len(y_test))
		writer.add_scalar('bert_'+'/network/'+'acc', acc, testep)

		testep+=1
		print(corrects)
		print(acc)

	tr_loss=tr_loss/float(len(x_train)/bs)
	print(tr_loss)

	

# torch.save(model.state_dict(),'trained_discriminator.pt')
# torch.save(model,'trained_discriminator_fullmodel.pt')

