import pandas as pd
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import fasttext 
from tqdm import tqdm 

def loadData():
    train_data = pd.read_csv('resources/NCBI_disease/tsv/train.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    return train_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    num_labels = len(label_list)
    label_list = [x for x in label_list if not pd.isna(x)]
    id2label = {0 : 'B', 1 : 'I' , 2 : 'O' }
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

class dataset(Dataset):
    def __init__(self, tokens, labels, ft_model, label2id):
        assert len(tokens) == len(labels)
        self.tokens = tokens
        self.labels = labels
        self.len = len(tokens)
        self.ft_model = ft_model
        self.label2id = label2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_vec = [0.0, 0.0, 0.0]
        label_vec[self.label2id[label]] = 1.0
        token = self.tokens[idx]
        # print(token)
        token_embedding = self.ft_model[token]

        return torch.tensor(token_embedding, dtype=torch.float32), torch.tensor(label_vec, dtype=torch.float32)


class LinearTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearTagger, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x 


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        # move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # calculate loss
        loss = criterion(outputs, labels)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss

def main():
    """
    A simple bias model is trained. It used fasttest word embedding,
    and a hidden layer for decision making.
    The model is trained and saved here.
    """
    batch_size = 64
    train_data = loadData()
    id2label,label2id = IdToLabelAndLabeltoId(train_data)

    # loading fasttext
    ft_model = fasttext.load_model('cc.en.300.bin')

    # removing NaN
    train_data = train_data.dropna()
    token_lst = []
    label_lst = []
    for index, row in train_data.iterrows():
        token_lst.append(row['Tokens'])
        label_lst.append(row['Labels'])

    train_dataset = dataset(token_lst, label_lst, ft_model, label2id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 300
    hidden_dim = 100
    output_dim = 3
    lr = 0.001
    #Initialize the model
    model = LinearTagger(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

    # saving the model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved!")

if __name__ == "__main__":
    main()