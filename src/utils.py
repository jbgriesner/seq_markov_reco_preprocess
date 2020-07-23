import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import fileinput

def concatenate_files(file, pattern):
    """
        If the ratings/checkins file exists for each dataset, we do nothing,
        otherwise we just concatenate the pieces of files
    """
    if not os.path.exists(file):
        checkins_files = glob.glob(pattern)
        with open(file, 'w') as out_file:
            input_lines = fileinput.input(checkins_files)
            for line in tqdm(input_lines):
                out_file.write(line)

def train_tune_test_split(clean_DIR, X, test_min_clicks):
    """
        Split the 'X' matrix into train, tune and test .tsv files
        in the given 'clean_DIR' folder.
    """
    train_file = f"{clean_DIR}/train.tsv"
    tune_file = f"{clean_DIR}/tune.tsv"
    test_file = f"{clean_DIR}/test.tsv"

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(tune_file):
        os.remove(tune_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    all_items = []
    with open(train_file, 'w') as train, open(tune_file, 'w') as tune, open(test_file, 'w') as test:
        user = 0
        for row in tqdm(X):
            if row.sum() < test_min_clicks:
                all_items.extend(row.nonzero()[1])
        for row in tqdm(X):
            items = row.nonzero()[1]
            np.random.shuffle(items)
            if row.sum() > test_min_clicks:
                splits = np.split(items, [int(.8 * len(items)), int(.9 * len(items))])
                for item in splits[0]:
                        train.write(f"{user}\t{item}\n")
                for item in splits[1]:
                    if item in all_items:
                        tune.write(f"{user}\t{item}\n")
                for item in splits[2]:
                    if item in all_items:
                        test.write(f"{user}\t{item}\n")
            else:
                for item in items:
                    train.write(f"{user}\t{item}\n")
            user += 1


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_rows(tp, min_uc=5, min_sc=5):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te
