import os 
import math
import pandas as pd
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import earlyStopping

input_path = 'datasets'
batch_size = 64


def read_data(dataset_name):
    train_path = os.path.join(input_path, dataset_name, 'train.txt')
    devel_path = os.path.join(input_path, dataset_name, 'devel.txt')
    test_path = os.path.join(input_path, dataset_name, 'test.txt')
    train_token_lst, train_label_lst = [], []
    with open(train_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                train_token_lst.append(math.nan)
                train_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            train_token_lst.append(a[0].strip())
            train_label_lst.append(a[1].strip())

    train_data = pd.DataFrame({'Tokens': train_token_lst, 'Labels': train_label_lst})

    devel_token_lst, devel_label_lst = [], []
    with open(devel_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                devel_token_lst.append(math.nan)
                devel_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            devel_token_lst.append(a[0].strip())
            devel_label_lst.append(a[1].strip())

    devel_data = pd.DataFrame({'Tokens': devel_token_lst, 'Labels': devel_label_lst})

    test_token_lst, test_label_lst = [], []
    with open(test_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                test_token_lst.append(math.nan)
                test_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            test_token_lst.append(a[0].strip())
            test_label_lst.append(a[1].strip())

    test_data = pd.DataFrame({'Tokens': test_token_lst, 'Labels': test_label_lst})

    return train_data, devel_data, test_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if isinstance(tok, float):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    return sent_list,label_list

def create_vocab(s_lst):
    """
    Create the vocabulary and maps each token to id
    """
    word_to_ix = {}
    for sent in s_lst:
        parts = sent.split(' ')
        for word in parts:
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)

    word_to_ix['UNK'] = len(word_to_ix)
    word_to_ix['PAD'] = len(word_to_ix)

    return word_to_ix 

class dataset(Dataset):
    def __init__(self, s_lst, l_lst, vocab, label2id):
        self.sent_lst = s_lst 
        self.label_lst = l_lst 
        self.vocab = vocab 
        self.label2id = label2id 

    def __len__(self):
        return len(self.sent_lst)

    def __getitem__(self, idx):
        return self.sent_lst[idx], self.label_lst[idx]

def collate_fn(batch, vocab, label2id):
    batch_sentences = []
    batch_tags = []
    for i in range(len(batch)):
        batch_sentences.append(batch[i][0])
        batch_tags.append(batch[i][1])

    batch_max_len = max([len(s.split(' ')) for s in batch_sentences])

    batch_data = vocab['PAD']*np.ones((len(batch_sentences), batch_max_len))
    batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

    # copy data to batch_data and batch_labels
    for i in range(len(batch_sentences)):
        cur_len = len(batch_sentences[i].split(' '))
        batch_data[i][:cur_len] = [vocab[s] if s in vocab.keys() else vocab['UNK'] for s in batch_sentences[i].split(' ')]
        batch_labels[i][:cur_len] = [label2id[l] for l in batch_tags[i].split(',')]
    
    # convert data to torch LongTensor
    batch_data = torch.LongTensor(batch_data)
    batch_labels = torch.LongTensor(batch_labels)

    return batch_data, batch_labels 

class bilstmTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, num_of_tags):
        super(bilstmTagger, self).__init__()

        #maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bi_lstm1 = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(lstm_hidden_dim*2, lstm_hidden_dim, batch_first=True, bidirectional=True)
        # *2 as output from bilstm is twice lstm_hidden_dim
        # one forward lstm and one backward lstm
        self.fc = nn.Linear(lstm_hidden_dim*2, num_of_tags)

    def forward(self, s):
        #apply the embedding layer that maps each token to its embedding
        s = self.embedding(s) # dim: batch_size x batch_max_len x embedding_dim

        #run LSTM along the sentences
        s, _ = self.bi_lstm1(s) # dim: batch_size x batch_max_len x lstm_hidden_dim*2
        s, _ = self.bi_lstm2(s) # dim: batch_size x batch_max_len x lstm_hidden_dim*2

        #reshape s so that each row contains one token
        s = s.reshape(-1, s.shape[2]) # dim: batch_size*batch_max_len x lstm_hidden_dim*2

        #apply the fully connected layer and obtain the output for each token
        s = self.fc(s) # dim: batch_size*batch_max_len x num_of_tags

        # using log_softmax for numerical stability 
        return F.log_softmax(s, dim=1) # dim: batch_size*batch_max_len x num_of_tags

def loss_fn(outputs, labels):
    """Writing custom loss function as torch.nn.loss will add loss from PAD tokens as well"""

    labels = labels.view(-1)
    #mask out 'PAD' tokens
    mask = (labels >= 0).float()
    #count number of tokens
    num_tokens = int(torch.sum(mask).item())
    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs) / num_tokens

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch[0]
        labels = batch[1]
        #move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        #zero the gradients
        optimizer.zero_grad()
        #forward pass
        outputs = model(inputs)
        #calculate loss
        loss = loss_fn(outputs, labels)
        #backward pass
        loss.backward()
        #update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def validation(model, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        inputs = batch[0]
        labels = batch[1]
        #move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # get inference from model
        outputs = model(inputs)
        #calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

    return total_loss

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--require_training', help='will train model if specified', action='store_true')
    parser.add_argument('--do_testset', help='Generate predictions for test set to check performance of model', action='store_true')
    # parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    #read dataset
    train_data, devel_data, test_data = read_data(args.dataset_name)
    #mapping of labels to id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    #get sentence + label list from dataset
    train_sent_lst, train_label_lst = convert_to_sentence(train_data)
    #create vocab and token to id mapping
    word_to_ix = create_vocab(train_sent_lst)

    #initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #parameters for model
    vocab_size = len(word_to_ix)
    embedding_dim = 50
    lstm_hidden_dim = 50
    num_of_tags = len(id2label)
    lr = 1e-3

    #initialize the model
    model = bilstmTagger(vocab_size, embedding_dim, lstm_hidden_dim, num_of_tags).to(device)

    if args.require_training:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopper = earlyStopping.EarlyStopper(patience=3, min_delta=1)
        num_epochs = 40
        #train dataloader
        train_dataset = dataset(train_sent_lst, train_label_lst, word_to_ix, label2id)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, word_to_ix, label2id))
        #development dataloader
        devel_sent_lst, devel_label_lst = convert_to_sentence(devel_data)
        devel_dataset = dataset(devel_sent_lst, devel_label_lst, word_to_ix, label2id)
        devel_dataloader = DataLoader(devel_dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, word_to_ix, label2id))
        
        for epoch in tqdm(range(num_epochs)):
            loss = train(model, train_dataloader, optimizer, device)
            devel_loss = validation(model, devel_dataloader, device)
            print(f'Epoch: {epoch+1}, Train Loss:{loss}, Validation Loss:{devel_loss}\n')
            if early_stopper.early_stop(devel_loss):
                model_name = 'bilstm_bias_' + args.dataset_name + '.pth'
                model_path = os.path.join('saved_weights', model_name)
                torch.save(model.state_dict(), model_path)
                print(f'Model saved at epoch {epoch}.\n')
                break

    
    # if args.do_testset:
        # create test dataloader
        # test_sent_lst, test_label_lst = convert_to_sentence(test_data)
        # test_dataset = dataset()



if __name__ == '__main__':
    main()