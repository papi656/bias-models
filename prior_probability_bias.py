import os
import pandas as pd
import csv
import argparse
from tqdm import tqdm


input_path = 'datasets'
output_path = 'resources'

def read_and_count(dataset_name):
    path = os.path.join(input_path, dataset_name, 'train.tsv')
    tag_counts = {}
    data = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, names=['Tokens', 'Labels'], skip_blank_lines=False)

    data = data.dropna()

    for i in range(data.shape[0]):
        token, label = data.iloc[i].values
        if token.lower() not in tag_counts:
            tag_counts[token.lower()] = {'B':0, 'I':0, 'O': 0}
            tag_counts[token.lower()][label] += 1
        else:
            tag_counts[token.lower()][label] += 1

    
    return tag_counts

def generate_probability_dist_for_tokens(dataset_name, tag_counts, output_file):
    i_path = os.path.join(input_path, dataset_name, 'train.tsv')
    data = pd.read_csv(i_path, sep='\t', quoting=csv.QUOTE_NONE, names=['Tokens', 'Labels'], skip_blank_lines=False)

    o_path = os.path.join(output_path, dataset_name, output_file)
    with open(o_path, 'w') as fh:
        for i in tqdm(range(data.shape[0])):
            token, label = data.iloc[i].values
            if isinstance(token, float): # means it is empty line
                fh.write('\n')
            else:
                b_cnt = tag_counts[token.lower()]['B']
                i_cnt = tag_counts[token.lower()]['I']
                o_cnt = tag_counts[token.lower()]['O']
                total_cnt = b_cnt + i_cnt + o_cnt
                fh.write(f'{b_cnt/total_cnt},{i_cnt/total_cnt},{o_cnt/total_cnt}\n')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    tag_counts = read_and_count(args.dataset_name)
    generate_probability_dist_for_tokens(args.dataset_name, tag_counts, args.output_file)


if __name__ == '__main__':
    main()

