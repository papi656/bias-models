import math
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
    devel_data = pd.read_csv('resources/NCBI_disease/tsv/devel.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    test_data = pd.read_csv('resources/NCBI_disease/tsv/test.tsv',sep='\t',names=["Tokens","Labels"],skip_blank_lines = False)
    return train_data,test_data,devel_data

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
        token = self.tokens[idx]
        label = self.labels[idx]
        label_vec = [0.0, 0.0, 0.0]
        if not isinstance(token, float):
            label_vec[self.label2id[label]] = 1.0
        
        if isinstance(token, float):
            token_embedding = torch.ones(300) * 100
        else:
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


def evaluate(model, dataloader, criterion, device, id2label):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    num_tokens = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs.to(device)
            labels.to(device)
            outputs = model(inputs)
            #loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # for comparing predicted labels for accuracy
            pred_id = torch.argmax(outputs, dim=1)
            gold_id = torch.argmax(labels, dim=1)
            num_tokens += len(pred_id)
            for i in range(len(pred_id)):
                if id2label[pred_id[i].item()] == id2label[gold_id[i].item()]:
                    running_corrects += 1

    print(f'Validation Loss: {running_loss}, Validation Accuracy: {running_corrects/num_tokens}')


def generate_probability_dist_for_tokens(model, dataloader, output_file, device):
    """
    Generate probability distribution assigned by bias model for each token
    """
    model.eval()
    sentence_probability = []
    single_sent_prob = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs.to(device)
            labels.to(device)
            outputs = model(inputs)
            for i in range(len(inputs)):
                if torch.sum(inputs[i]).item() == 30000:
                    sentence_probability.append(single_sent_prob)
                    single_sent_prob = []
                else:
                    single_sent_prob.append(outputs[i])
        
    with open(output_file, "w") as fh:
        for sent_prob in sentence_probability:
            for tok_prob in sent_prob:
                fh.write(f'{tok_prob[0]},{tok_prob[1]},{tok_prob[2]}\n')
                # fh.write("\n")
            fh.write("\n")
        


def generate_prediction_file(model, dataloader, output_file, device, token_lst, id2label):
    """
    Generate prediction file with BIO tags for each token
    """
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs.to(device)
            labels.to(device)
            outputs = model(inputs)
            pred_id = torch.argmax(outputs, dim=1)
            batch_pred_labels = [id2label[id.item()] for id in pred_id]
            pred_labels.extend(batch_pred_labels)

    with open(output_file, 'w') as fh:
        for token, label in zip(token_lst, pred_labels):
            if isinstance(token, float):
                fh.write('\n')
            else:
                fh.write(f'{token}\t{label}\n')
    

def main():
    # Hyperparameters
    batch_size = 64

    train_data,test_data,devel_data = loadData()
    id2label,label2id = IdToLabelAndLabeltoId(train_data)

    # loading fasttext
    ft_model = fasttext.load_model('cc.en.300.bin')

    # Getting tokens and labels in separate lists
    train_token_lst = []
    train_label_lst = []
    for index, row in train_data.iterrows():
        train_token_lst.append(row['Tokens'])
        train_label_lst.append(row['Labels'])
    dev_token_lst = []
    dev_label_lst = []
    for index, row in devel_data.iterrows():
        dev_token_lst.append(row['Tokens'])
        dev_label_lst.append(row['Labels'])
    test_token_lst = []
    test_label_lst = []
    for index, row in devel_data.iterrows():
        test_token_lst.append(row['Tokens'])
        test_label_lst.append(row['Labels'])

    # Creating Dataloader
    train_dataset = dataset(train_token_lst, train_label_lst, ft_model, label2id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dev_dataset = dataset(dev_token_lst, dev_label_lst, ft_model, label2id)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = dataset(test_token_lst, test_label_lst, ft_model, label2id)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 300
    hidden_dim = 100
    output_dim = 3
    lr = 0.001
    #Initialize the model
    model = LinearTagger(input_dim, hidden_dim, output_dim).to(device)
    # loading pretrained weights for our model
    model.load_state_dict(torch.load('saved_weights/model.pth', map_location=torch.device('cpu')))
    criterion = nn.CrossEntropyLoss()
    
    generate_probability_dist_for_tokens(model, train_dataloader, "prob_dist.txt", device)
    generate_prediction_file(model, train_dataloader, "model_pred.txt", device, train_token_lst, id2label)
    evaluate(model, dev_dataloader, criterion, device, id2label)

if __name__ == "__main__":
    main()