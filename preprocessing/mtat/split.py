# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
import fire


class Preprocess:
    def __init__(self):
        print('start preprocessing')

    def get_csv_dataframe(self):
        self.csv_df = pd.read_csv(os.path.join(self.data_path, 'metadata/annotations_final.csv'), header=None, index_col=None, sep='\t')

    def get_top50_tags(self):
        tags = list(self.csv_df.loc[0][1:-1])
        tag_count = [np.array(self.csv_df[i][1:], dtype=int).sum() for i in range(1, 189)]
        top_50_tag_index = np.argsort(tag_count)[::-1][:50]
        top_50_tags = np.array(tags)[top_50_tag_index]
        np.save(open(os.path.join(self.data_path, 'data/tags.npy'), 'wb'), top_50_tags)
        return top_50_tag_index

    def write_tags(self, top_50_tag_index):
        binary = np.zeros((25863, 50))
        titles = []
        idx = 0
        for i in range(1, 25864):
            features = np.array(self.csv_df.loc[i][top_50_tag_index+1], dtype=int)
            title = self.csv_df.loc[i][189]
            #if np.sum(features) != 0:
            binary[idx] = features
            idx += 1
            titles.append(title)

        binary = binary[:len(titles)]
        np.save(open(os.path.join(self.data_path, 'data/binary.npy'), 'wb'), binary)
        return titles, binary

    def split(self, titles, binary):
        tr = []
        tr_new = []
        val = []
        val_new = []
        test = []
        test_new = []
        for i, title in enumerate(titles):
            if int(title[0], 16) < 12:
                tr.append(str(i)+'\t'+title)
                if binary[i].sum() > 0:
                    tr_new.append(str(i)+'\t'+title)
            elif int(title[0], 16) < 13:
                val.append(str(i)+'\t'+title)
                if binary[i].sum() > 0:
                    val_new.append(str(i)+'\t'+title)
            else:
                test.append(str(i)+'\t'+title)
                if binary[i].sum() > 0:
                    test_new.append(str(i)+'\t'+title)
        np.save(open(os.path.join(self.data_path, 'data/train.npy'), 'wb'), tr)
        np.save(open(os.path.join(self.data_path, 'data/valid.npy'), 'wb'), val)
        np.save(open(os.path.join(self.data_path, 'data/test.npy'), 'wb'), test)
        np.save(open(os.path.join(self.data_path, 'data/train_new.npy'), 'wb'), tr_new)
        np.save(open(os.path.join(self.data_path, 'data/valid_new.npy'), 'wb'), val_new)
        np.save(open(os.path.join(self.data_path, 'data/test_new.npy'), 'wb'), test_new)

    def run(self, data_path):
        self.data_path = data_path
        self.get_csv_dataframe()
        top_50_tag_index = self.get_top50_tags()
        titles, binary = self.write_tags(top_50_tag_index)
        self.split(titles, binary)


if __name__ == '__main__':
    p = Preprocess()
    fire.Fire({'run': p.run})
