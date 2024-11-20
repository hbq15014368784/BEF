import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
import base_model
from base_model import GenB, Discriminator

from train import train
import utils
import click

from utils1.proco import ProCoLoss


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1","RAD"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--eval_each_epoch', default=True,help="Evaluate every epoch, instead of at the end")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/nvidiaexp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=114514, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default=None)

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=True,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', default='data_RAD', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm('Exp directory already exists in {}. Erase?'
                                 .format(args.output, default=False)):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)

        else:
            if args.load_checkpoint_path is None:
                os._exit(1)


    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    elif dataset=='RAD':
        dictionary = Dictionary.load_from_file('data_RAD/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', args, dictionary, dataset=dataset,
                                   cache_image_features=args.cache_features)

    if dataset=='RAD':
        print("Building test dataset...")
        eval_dset = VQAFeatureDataset('test', args, dictionary, dataset=dataset,
                                      cache_image_features=args.cache_features)
    else:
        print("Building test dataset...")
        eval_dset = VQAFeatureDataset('val', args, dictionary, dataset=dataset,
                                      cache_image_features=args.cache_features)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    genb = GenB(num_hid=1024, dataset=train_dset).cuda()
    discriminator = Discriminator(num_hid=1024, dataset=train_dset).cuda()
    if dataset=='cpv1':
        model.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
        genb.w_emb.init_embedding('data/glove6b_init_300d_v1.npy')
    elif dataset=='cpv2' or dataset=='v2':
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        genb.w_emb.init_embedding('data/glove6b_init_300d.npy')

    if dataset=='RAD':
        qid2type = None
    else:
        with open('util/qid2type_%s.json'%args.dataset,'r') as f:
            qid2type=json.load(f)

    if args.load_checkpoint_path is not None:
        ckpt = torch.load(os.path.join('logs', args.load_checkpoint_path, 'model.pth'))
        model_dict = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(model_dict)
        model.load_state_dict(model_dict)

    model=model.cuda()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)

    print("Starting training...")

    # define prco loss
    # criterion_scl = ProCoLoss(contrast_dim=1024, temperature=1.0, num_classes=train_dset.num_ans_candidates).cuda(0).double()

    train(model, genb, discriminator, train_loader, eval_loader, args, qid2type)

if __name__ == '__main__':
    main()
