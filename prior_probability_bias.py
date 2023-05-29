import os
import argparse


input_path = 'datasets'
output_path = 'resources'

def read_and_count(dataset_name):
    file_path = os.path.join(input_path, dataset_name, 'train.txt')
    tag_counts = {}
    token_lst = []
    with open(file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                token_lst.append("")
                continue 
            parts = line.split('\t')
            token, label = parts[0].strip(), parts[1].strip()
            token_lst.append(token)
            if token.lower() not in tag_counts:
                tag_counts[token.lower()] = {'B':0, 'I':0, 'O':0}
                tag_counts[token.lower()][label] += 1
            else:
                tag_counts[token.lower()][label] += 1

    return tag_counts, token_lst

def generate_probability_dist_for_tokens(dataset_name, tag_counts, token_lst):

    output_file_path = os.path.join(output_path, dataset_name, 'BC5CDR_prior_prob.txt')
    with open(output_file_path, 'w') as fh:
        for tok in token_lst:
            if len(tok.strip()) == 0:
                fh.write(f'\n')
            else:
                b_cnt = tag_counts[tok.lower()]['B']
                i_cnt = tag_counts[tok.lower()]['I']
                o_cnt = tag_counts[tok.lower()]['O']
                total_cnt = b_cnt + i_cnt + o_cnt 
                fh.write(f'{b_cnt/total_cnt},{i_cnt/total_cnt},{o_cnt/total_cnt}\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)

    args = parser.parse_args()

    tag_counts, token_lst = read_and_count(args.dataset_name)
    generate_probability_dist_for_tokens(args.dataset_name, tag_counts, token_lst)

if __name__ == '__main__':
    main()