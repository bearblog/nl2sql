# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse
from contextlib import contextmanager


def model_config():
    parser = argparse.ArgumentParser()
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data")
    data_arg.add_argument("--model_dir", type=str, default="./model")
    data_arg.add_argument("--log_dir", type=str, default="./code/log_bad_cases")
    data_arg.add_argument("--submit_dir", type=str, default="./submit")
    data_arg.add_argument("--log_path", type=str, default="./code/log_bad_cases/nl2sql.log")

    config = parser.parse_args()
    return config


def init_logger(log_dir):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    # 控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    # 日志文件
    handler_file = logging.FileHandler(os.path.join(log_dir, "nl2sql.log"))
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger


@contextmanager
def timer(msg, prefix=""):
    t0 = time.time()
    print(f'{prefix}[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'{prefix}[{msg}] done in {elapsed_time / 60:.2f} min.\n')


if __name__ == '__main__':
    logger = get_train_logger()
