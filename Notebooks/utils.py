import numpy as np


def get_list_files(train_file,val_file):

    train_list = []

    with open(train_file) as f:
        lines = f.readlines()

        for line in lines:
            row = line.strip().split()[0]
            train_list.append(row)

    val_list = []

    with open(val_file) as f:
        lines = f.readlines()

        for line in lines:
            row = line.strip().split()[0]
            val_list.append(row)
            
    return train_list,val_list