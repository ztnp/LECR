# -*- coding:utf-8 -*-
import csv
import random
import pandas as pd

positive_data_filepath = './positive_data.csv'

negative_data_filepath = './negative_data.csv'

train_set_filepath = './lecr_dataset.csv'


def load_data():
    # correlations
    correlations_dict = dict()
    correlations_filepath = './dataset/correlations.csv'
    with open(correlations_filepath, 'r') as f:
        for line in f:
            lines = line.strip().split(',')
            correlations_dict.update({lines[0]: lines[1]})
    correlations_dict.pop('topic_id')

    # topics
    topics_dict = dict()
    topics_filepath = './dataset/topics.csv'
    with open(topics_filepath, newline='') as csvfile:
        datas = csv.reader(csvfile)
        for lines in datas:
            topics_dict.update({lines[0]: lines})
    topics_dict.pop('id')

    # content
    content_dict = dict()
    content_filepath = './dataset/content.csv'
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
    random_num = 12
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
    positive_data_df = pd.read_csv(positive_data_filepath, header=None)
    positive_data_df[2] = [random.uniform(0.75, 0.90) for _ in range(len(positive_data_df))]

    negative_data_df = pd.read_csv(negative_data_filepath, header=None)
    negative_data_df[2] = [random.uniform(0, 0.40) for _ in range(len(negative_data_df))]

    # result = positive_data_df.append(negative_data_df)
    result = pd.concat([positive_data_df, negative_data_df])
    result = result.sample(frac=1)
    result.to_csv(train_set_filepath, header=False)





def main():
    # correlations_dict, topics_dict, content_dict = load_dataset()
    # generate_positive_data(correlations_dict, topics_dict, content_dict)
    # generate_negative_data(correlations_dict, topics_dict, content_dict)
    pass


if __name__ == '__main__':
    main()
