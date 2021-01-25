import csv
import torch

from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

def load(opt, splits):
    """
    Load the TUF train dataset
    :return:
    """
    filename = opt["data.tuf_filename"]
    labels_to_idx = opt["data.labels_to_idx"]
    ret = {}

    train_data, n_way, n_shot = loadtuf_csv(filename=filename, labels_to_idx=labels_to_idx)

    assert n_way == opt['data.way'], "Invalid number of n_way"
    assert n_shot == opt['data.shot'], "Invalid number of n_shot"

    for split in splits:

        #if split in ['val', 'test'] and opt['data.test_way'] != 0:
        #    n_way = #opt['data.test_way']
        #else:
        #    n_way = opt['data.way']
        n_way = len(labels_to_idx)

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(train_data))
        else:
            sampler = EpisodicBatchSampler(n_classes=n_shot,
                                           n_way=n_way, n_episodes=n_episodes)

        ret[split] = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=0)

    return ret


def loadtuf_csv(filename, labels_to_idx):
    """
    Load the CSV file
    """
    train = []
    with open(filename, 'r', newline='\n') as fh:
        reader = csv.reader(fh, delimiter=',')

        n_way = 0
        n_shot = 0
        for row in reader:

            if '#' in row:
                n_way = int(row[1].split(':')[-1])
                n_shot = int(row[2].split(':')[-1])
            else:

                label = row[2]
                train.append([float(row[0]), float(row[1]), labels_to_idx[label]])
    return train, n_way, n_shot
