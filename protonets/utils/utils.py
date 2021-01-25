import json
import argparse


def read_config_file(filename):
    """
    Read the json configuration file and
    return a map with the config entries
    """
    with open(filename) as json_file:
        json_input = json.load(json_file)
        return json_input


def build_parser_from_config_map(config_map):
    """
    Build the parser arguments from the configuration map
    """

    parser = argparse.ArgumentParser(description='Train prototypical networks')

    # data args
    #default_dataset = 'omniglot'
    parser.add_argument('--data.dataset', type=str, default=config_map['--data.dataset'], #default_dataset,
                        metavar='DS', help="data set name (default: {:s})".format(config_map['--data.dataset']))

    if config_map['--data.dataset'] == 'tuf':
        parser.add_argument('--data.tuf_filename', type=str, default=config_map['--data.tuf_filename'],  # default_dataset,
                            metavar='DS', help="TUF data set name")

        parser.add_argument('--data.labels_to_idx', type=map, default=config_map['--data.labels_to_idx'],
                            # default_dataset,
                            metavar='DS', help="Labels TUF map")


    default_split = 'vinyals'
    parser.add_argument('--data.split', type=str, default=config_map['--data.split'], #default_split,
                        metavar='SP', help="split name (default: {:s})".format(config_map['--data.split']))

    parser.add_argument('--data.way', type=int, default=config_map['--data.way'], #,60,
                        metavar='WAY', help="number of classes per episode")

    parser.add_argument('--data.shot', type=int, default=config_map['--data.shot'], #5,
                        metavar='SHOT', help="number of support examples per class")

    parser.add_argument('--data.query', type=int, default=config_map['--data.query'], #5,
                        metavar='QUERY', help="number of query examples per class")

    parser.add_argument('--data.test_way', type=int, default=config_map['--data.test_way'], #5,
                        metavar='TESTWAY', help="number of classes per episode in test.")

    parser.add_argument('--data.test_shot', type=int, default=config_map['--data.test_shot'], #0,
                        metavar='TESTSHOT', help="number of support examples per class in test.")

    parser.add_argument('--data.test_query', type=int, default=config_map['--data.test_query'], #15,
                        metavar='TESTQUERY', help="number of query examples per class in test.")

    parser.add_argument('--data.train_episodes', type=int, default=config_map['--data.train_episodes'], #100,
                        metavar='NTRAIN', help="number of train episodes per epoch.")

    parser.add_argument('--data.test_episodes', type=int, default=config_map['--data.test_episodes'], #100,
                        metavar='NTEST', help="number of test episodes per epoch.")

    parser.add_argument('--data.trainval', action=config_map['--data.trainval'], #'store_true',
                        help="run in train+validation mode.")

    parser.add_argument('--data.sequential', action=config_map['--data.sequential'], #'store_true',
                        help="use sequential sampler instead of episodic.")

    parser.add_argument('--data.cuda', action=config_map['--data.cuda'], #'store_true',
                        help="run in CUDA mode.")

    # model args
    parser.add_argument('--model.model_name', type=str, default=config_map['--model.model_name'], #default_model_name,
                        metavar='MODELNAME', help="model name.")

    parser.add_argument('--model.x_dim', type=str, default=config_map['--model.x_dim'], #'1,28,28',
                        metavar='XDIM', help="dimensionality of input.")

    parser.add_argument('--model.hid_dim', type=int, default=config_map['--model.hid_dim'], #64,
                        metavar='HIDDIM', help="dimensionality of hidden layers")

    parser.add_argument('--model.z_dim', type=int, default=config_map['--model.z_dim'], #64,
                        metavar='ZDIM', help="dimensionality of input images.")

    # train args
    parser.add_argument('--train.epochs', type=int, default=config_map['--train.epochs'], #10000,
                        metavar='NEPOCHS', help='number of epochs to train.')

    parser.add_argument('--train.optim_method', type=str, default=config_map['--train.optim_method'], #'Adam',
                        metavar='OPTIM', help='optimization method')

    parser.add_argument('--train.learning_rate', type=float, default=config_map['--train.learning_rate'], #0.001,
                        metavar='LR', help='learning rate.')

    parser.add_argument('--train.decay_every', type=int, default=config_map['--train.decay_every'], #20,
                        metavar='LRDECAY', help='number of epochs after which to decay the learning rate')

    default_weight_decay = 0.0
    parser.add_argument('--train.weight_decay', type=float, default=config_map['--train.weight_decay'], #default_weight_decay,
                        metavar='WD', help="weight decay")

    parser.add_argument('--train.patience', type=int, default=config_map['--train.patience'], #200,
                        metavar='PATIENCE',
                        help='number of epochs to wait before validation improvement')

    # log args
    default_fields = 'loss,acc'
    parser.add_argument('--log.fields', type=str, default=config_map['--log.fields'], #default_fields,
                        metavar='FIELDS',
                        help="fields to monitor during training ")

    default_exp_dir = 'results'
    parser.add_argument('--log.exp_dir', type=str, default=config_map['--log.exp_dir'], #default_exp_dir,
                        metavar='EXP_DIR',
                        help="directory where experiments should be saved")

    args = vars(parser.parse_args())
    return args
