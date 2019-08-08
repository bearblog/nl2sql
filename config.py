#coding=utf-8

from contextlib import contextmanager
import time
import logging
import sys

def get_train_logger():
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    #控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    #日志文件
    handler_file = logging.FileHandler('train.log')
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')
# with timer('标注数据'):
#     #V,v,NPC = voteforLabels(G,300)
#     print("a")



if __name__ == '__main__':
    # with timer('标注数据'):
    #     #V,v,NPC = voteforLabels(G,300)
    #     print("a")
    logger = get_train_logger()
    for i in range(10):
        # print ('Epoch %d'%(i+1))
        logger.info('starting epoch {}'.format(i))
