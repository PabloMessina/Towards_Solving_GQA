import os
import sys
import json
import pickle
import argparse
import nltk
import tqdm
from PIL import Image

def save_as_pickle(data, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'wb') as f:
        pickle.dump(data, f)
        print('sucessfully saved data to %s' % dst_path)

def process_questions(data_dir, output_dir, split, word2index, answer2index):

    questions_path = os.path.join(data_dir, 'questions',
        'CLEVR_{}_questions.json'.format(split))    
    with open(questions_path) as f:
        print('loading questions from %s ...' % questions_path)
        data = json.load(f)

    processed_questions = []
    word_index = 1
    answer_index = 0
    print('processing questions ...')
    for question in tqdm.tqdm(data['questions']):

        # tokenize question and map each token its index
        words = nltk.word_tokenize(question['question'])
        q_token_indexes = []
        for word in words:
            try:
                idx = word2index[word]
            except KeyError:
                idx = word2index[word] = word_index
                word_index += 1
            q_token_indexes.append(idx)

        # map answer to its index
        answer_word = question['answer']
        try:
            a_index = answer2index[answer_word]
        except:
            a_index = answer2index[answer_word] = answer_index
            answer_index += 1

        # append processed question
        processed_questions.append((
            question['image_filename'],
            q_token_indexes, a_index,
            question['question_family_index']))

    save_as_pickle(
        processed_questions,
        os.path.join(output_dir, '%s.pkl' % split))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    word2index = dict()
    answer2index = dict()
    process_questions(args.data_root_dir, args.output_dir, 'train', word2index, answer2index)
    process_questions(args.data_root_dir, args.output_dir, 'val', word2index, answer2index)
    save_as_pickle(dict(
        word2index=word2index,
        answer2index=answer2index,
    ), os.path.join(args.output_dir, 'dic.pkl'))