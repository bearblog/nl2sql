import torch
# from sqlnet.utils import *
from utils.utils import *
# from sqlnet.model.sqlnet import SQLNet
from models.sqlnet import SQLNet
import logging
import os
import argparse
import sys


def get_train_logger(args):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    #控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    #日志文件
    handler_file = logging.FileHandler('{}.log'.format(os.path.join(args.dout,args.nick)))
    # formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger
def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu to train')
    parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention')
    parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet')
    parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    parser.add_argument('--logdir', type=str, default='log', help='Path of save experiment log')
    parser.add_argument('--mode', type=str, default='train', help='train,adjust,test,rl')
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--emb_size', type=int, default=300, help='embedding size')
    parser.add_argument('--dexp', help='root experiment folder', default='result')
    parser.add_argument('--model', help='which model to use', default='sqlnet', choices=get_models())
    parser.add_argument('-n', '--nick', help='nickname for model', default='sample1000')
    parser.add_argument('--emb_path', help='embedding path', default='char_embedding.json')
    parser.add_argument('--output_dir', type=str, default='predict', help='Output path of prediction result')

    args = parser.parse_args()
    # args.dout = os.path.join(args.dexp,args.model,args.nick)
    args.dout = os.path.join(args.dexp,args.model)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    return args
def train(args,use_small):
    n_word = args.emb_size
    gpu = args.gpu
    batch_size =args.bs
    learning_rate = args.lr

    # load dataset
    train_sql, train_table, train_db, dev_sql, dev_table, dev_db = load_dataset(use_small=use_small)
    word_emb = load_word_emb('data/{}'.format(args.emb_path))
    model = SQLNet(word_emb, N_word=n_word, use_ca=args.ca, gpu=gpu, trainable_emb=args.train_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    # used to record best score of each sub-task
    '''
    sn  select_number
    sc  select_colmun
    sa  select_agg
    wn  where_number
    wc  where_column
    wo  where-operator
    wv  where_value
    wr  where-relationship
    '''
    best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    best_lf, best_lf_idx = 0.0, 0
    best_ex, best_ex_idx = 0.0, 0
    print ("#"*20+"  Star to Train  " + "#"*20)
    logger = get_train_logger(args)
    for i in range(args.epoch):
        print ('Epoch %d'%(i+1))
        logger.info('starting epoch {}'.format(i))
        # train on the train dataset
        train_loss = epoch_train(model, optimizer, batch_size, train_sql, train_table)
        # evaluate on the dev dataset
        dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db)
        # accuracy of each sub-task
        # print ('Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f'%(
        #     dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6], dev_acc[0][7]))
        logger.info('Sel-Num: %.3f, Sel-Col: %.3f, Sel-Agg: %.3f, W-Num: %.3f, W-Col: %.3f, W-Op: %.3f, W-Val: %.3f, W-Rel: %.3f'%(
            dev_acc[0][0], dev_acc[0][1], dev_acc[0][2], dev_acc[0][3], dev_acc[0][4], dev_acc[0][5], dev_acc[0][6], dev_acc[0][7]))

        # save the best model
        if dev_acc[1] > best_lf:
            best_lf = dev_acc[1]
            best_lf_idx = i + 1
            # torch.save(model.state_dict(), 'saved_model/best_model')
            # torch.save(model.state_dict(), 'saved_model/{}'.format(args.model))
            torch.save(model.state_dict(), '{}/{}'.format(args.dout,args.nick))
        if dev_acc[2] > best_ex:
            best_ex = dev_acc[2]
            best_ex_idx = i + 1

        # record the best score of each sub-task
        if True:
            if dev_acc[0][0] > best_sn:
                best_sn = dev_acc[0][0]
                best_sn_idx = i+1
            if dev_acc[0][1] > best_sc:
                best_sc = dev_acc[0][1]
                best_sc_idx = i+1
            if dev_acc[0][2] > best_sa:
                best_sa = dev_acc[0][2]
                best_sa_idx = i+1
            if dev_acc[0][3] > best_wn:
                best_wn = dev_acc[0][3]
                best_wn_idx = i+1
            if dev_acc[0][4] > best_wc:
                best_wc = dev_acc[0][4]
                best_wc_idx = i+1
            if dev_acc[0][5] > best_wo:
                best_wo = dev_acc[0][5]
                best_wo_idx = i+1
            if dev_acc[0][6] > best_wv:
                best_wv = dev_acc[0][6]
                best_wv_idx = i+1
            if dev_acc[0][7] > best_wr:
                best_wr = dev_acc[0][7]
                best_wr_idx = i+1
        # print ('Train loss = %.3f' % train_loss)
        # print ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))
        # print ('Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx))
        # print ('Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx))
        logger.info ('Train loss = %.3f' % train_loss)
        logger.info ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))
        logger.info ('Best Logic Form: %.3f at epoch %d' % (best_lf, best_lf_idx))
        logger.info ('Best Execution: %.3f at epoch %d' % (best_ex, best_ex_idx))
        if (i+1) % 10 == 0:
            logger.info ('Best val acc: %s\nOn epoch individually %s'%(
                    (best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv),
                    (best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx)))

def test(args,use_small):
    
    n_word = args.emb_size
    gpu = args.gpu
    batch_size =args.bs
    learning_rate = args.lr
    dev_sql, dev_table, dev_db, test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')
    word_emb = load_word_emb('data/{}'.format(args.emb_path))
    model = SQLNet(word_emb, N_word=n_word, use_ca=args.ca, gpu=gpu, trainable_emb=args.train_emb)
    model_path = '{}/{}'.format(args.dout,args.nick)
    logger = get_train_logger(args)
    logger.info("Loading from %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    logger.info("Loaded from %s" % model_path)
    dev_acc = epoch_acc(model, batch_size, dev_sql, dev_table, dev_db)
    logger.info ('Dev Logic Form Accuracy: %.3f, Execution Accuracy: %.3f' % (dev_acc[1], dev_acc[2]))

    logger.info ("Start to predict test set")
    predict_path = '{}/{}.json'.format(args.output_dir,'_'.join([args.model,args.nick]))
    print(predict_path)
    predict_test(model, batch_size, test_sql, test_table, predict_path)
    logger.info ("Output path of prediction result is %s" % args.output_dir)






def main(args):
    

    if args.mode == 'train':
        if args.toy:
            use_small=True
        else:
            use_small=False
            # load dataset
        train(args,use_small)
        test(args,use_small)
    if args.mode == 'test':
        if args.toy:
            use_small=True
        else:
            use_small=False
            # load dataset
        test(args,use_small)

        
        # m.load_glove_embedding()
        # m.train()
    # elif args.mode == 'adjust':
    #     m.load_model()
    #     m.train()
    # elif args.mode == 'test':
    #     m.load_model()
    #     m.eval()
    # elif args.mode == 'rl':
    #     m.load_model()
    #     m.reinforce_tune()


if __name__ == '__main__':
    args = get_args()
    main(args)