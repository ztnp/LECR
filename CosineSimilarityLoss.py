# -*- coding:utf-8 -*-
import csv
import os.path
import random
import time

from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from torch import nn
from torch.utils.data import DataLoader

random_num = 12

epochs = 10
batch_size = 32
num_workers = 10
prefetch_factor = batch_size

warmup_steps = 200
evaluation_steps = 10000

topics_filepath = './dataset/topics.csv'
content_filepath = './dataset/content.csv'
correlations_filepath = './dataset/correlations.csv'

train_set_filepath = './lecr_dataset.csv'
positive_data_filepath = './positive_data.csv'
negative_data_filepath = './negative_data.csv'

save_filepath = './output_st'
if not os.path.exists(save_filepath):
    os.mkdir(save_filepath)

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(random.randint(0, 3))


def timer(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)

        msg = 'function {} runtime: {}s'.format(func.__name__, time.time() - local_time)
        print(msg)
        return res

    return wrapper


def load_data():
    # correlations
    correlations_dict = dict()
    with open(correlations_filepath, 'r') as f:
        for line in f:
            lines = line.strip().split(',')
            correlations_dict.update({lines[0]: lines[1]})
    correlations_dict.pop('topic_id')

    # topics
    topics_dict = dict()
    with open(topics_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        for lines in datas:
            topics_dict.update({lines[0]: lines})
    topics_dict.pop('id')

    # content
    content_dict = dict()
    with open(content_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        for lines in datas:
            content_dict.update({lines[0]: lines})
    content_dict.pop('id')

    return correlations_dict, topics_dict, content_dict


def generate_positive_data(correlations_dict, topics_dict, content_dict):
    with open(positive_data_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k, v in correlations_dict.items():
            topic_datas = topics_dict.get(k)
            topic_data = '{}'.format(topic_datas[1])

            v_list = v.split(' ')
            for item in v_list:
                content_datas = content_dict.get(item)
                content_data = '{}'.format(content_datas[1])
                writer.writerow([topic_data, content_data])


def generate_negative_data(correlations_dict, topics_dict, content_dict):
    with open(negative_data_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        content_ids = list(set(content_dict.keys()))

        for k, v in topics_dict.items():

            if v[-1] == 'False':
                continue

            content_id_sample = []

            while True:
                content_id = random.choice(content_ids)
                if content_id in content_id_sample:
                    continue

                correlations_content = correlations_dict.get(k)
                if content_id in correlations_content:
                    # if any([content_id == x for x in correlations_content.split(' ')]):
                    continue
                content_id_sample.append(content_id)
                if len(content_id_sample) > random_num:
                    break

            for id_item in content_id_sample:
                content_datas = content_dict.get(id_item)
                writer.writerow([v[1], content_datas[1]])


def generate_data():
    csvfile_result = open(train_set_filepath, 'w', newline='')
    writer = csv.writer(csvfile_result)

    with open(positive_data_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        for lines in datas:
            writer.writerow(lines + [str(random.uniform(0.75, 0.90))])

    with open(negative_data_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        for lines in datas:
            writer.writerow(lines + [str(random.uniform(0, 0.40))])

    csvfile_result.close()


def load_dataset():
    line_content = []
    with open(train_set_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        # line_content = list(datas)
        for item in datas:
            item[2] = float(item[2])
            line_content.append(item)

    for _ in range(20):
        index = [i for i in range(len(line_content))]
        random.shuffle(index)
        line_content = [line_content[x] for x in index]

    lf = len(line_content)
    test_dataset = line_content[:int(lf * 0.02)]
    train_dataset = line_content[int(lf * 0.02):]

    return train_dataset, test_dataset


@timer
def train_model():
    word_embedding_model = models.Transformer('bert-base-multilingual-cased', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=512,
        activation_function=nn.Tanh())
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model])

    # load dataset
    train_dataset, test_dataset = load_dataset()
    train_examples = list(map(lambda x: InputExample(texts=[x[0], x[1]], label=float(x[2])), train_dataset))

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    # define loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # define evaluator
    sentences1, sentences2, scores = list(zip(*test_dataset))
    # scores = [0 if z < 0.86 else 1 for z in scores]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(list(sentences1), list(sentences2), scores)

    # train model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        # scheduler='warmupcosine',
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        output_path=save_filepath,
        save_best_model=True)
    print('finish')


def main():
    # correlations_dict, topics_dict, content_dict = load_data()
    # generate_positive_data(correlations_dict, topics_dict, content_dict)
    # generate_negative_data(correlations_dict, topics_dict, content_dict)

    train_model()


if __name__ == '__main__':
    main()
