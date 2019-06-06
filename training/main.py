import os
import argparse
from solver import Solver

def main(config):
    assert config.mode in {'train', 'test'},\
        'invalid mode: "{}" not in ["train", "test"]'.format(config.mode)
    assert config.dataset in {'mtat', 'msd'},\
        'invalid mode: "{}" not in ["mtat", "msd"]'.format(config.task)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if config.dataset == 'msd':
        from data_loader.msd_loader import get_audio_loader
    elif config.dataset == 'mtat':
        from data_loader.mtat_loader import get_audio_loader

    if config.mode == 'train':
        data_loader = get_audio_loader(config.data_path,
                                       config.batch_size,
                                       trval='TRAIN',
                                       num_workers=config.num_workers,
                                       dataset=config.dataset,
                                       data_type=config.data_type,
                                       input_length=config.input_length)
    elif config.mode == 'test':
        data_loader = get_audio_loader(config.data_path,
                                       config.batch_size,
                                       trval='TEST',
                                       num_workers=config.num_workers,
                                       dataset=config.dataset,
                                       data_type=config.data_type,
                                       input_length=config.input_length)

    solver = Solver(data_loader, config)

    if config.mode == 'train':
        print('train')
        solver.train()
    elif config.mode == 'test':
        print('test')
        solver.test()
    else:
        print('invalid mode')
        exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyper-parameters
    parser.add_argument('--conv_channels', type=int, default=64)
    parser.add_argument('--attention_channels', type=int, default=512)
    parser.add_argument('--attention_layers', type=int, default=2)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=50)
    parser.add_argument('--input_length', type=int, default=256)
    parser.add_argument('--attention_length', type=int, default=271)
    parser.add_argument('--attention_dropout', type=float, default=0.4)
    parser.add_argument('--fc_dropout', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd'])
    parser.add_argument('--data_type', type=str, default='spec', choices=['spec', 'raw'])

    # Training settings
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--architecture', type=str, default='pons_won', choices=['pons_pons', 'pons_won', 'lee_won', 'lee_lee'])

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=int, default=1)

    parser.add_argument('--model_save_path', type=str, default='./../models')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')

    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--model_save_epoch', type=int, default=1)

    parser.add_argument('--use_nsml', action='store_true')

    config = parser.parse_args()
    if config.use_nsml:
        from nsml import DATASET_PATH
        config.data_path = DATASET_PATH

    print(config)
    main(config)
