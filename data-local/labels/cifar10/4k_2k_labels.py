#!coding:utf-8
from collections import defaultdict
import random

if __name__ == '__main__':

    k4_dir = '4000_balanced_labels'
    k2_dir = '2000_balanced_labels'
    for i in range(20):
        k4_path = '{}/{:02d}.txt'.format(k4_dir, i)
        k4_dict = defaultdict(list)
        with open(k4_path, 'r') as k4_file:
            for line in k4_file:
                img, label = line.split()
                k4_dict[label].append(img)
        for key, val in k4_dict.items():
            print('{}: {}'.format(key, len(val)))

        each_labeled_n = 200
        k2_path = '{}/{:02d}.txt'.format(k2_dir, i)
        k2_list = []
        for label, k4_each_list in k4_dict.items():
            imgs = random.sample(k4_each_list, each_labeled_n)
            for img in imgs:
                k2_list.append('{} {}\n'.format(img, label))

        labeled_index = list(range(len(k2_list)))
        #random.shuffle(labeled_index)

        with open(k2_path, 'w') as k2_file:
            for i in labeled_index:
                k2_file.write(k2_list[i])
