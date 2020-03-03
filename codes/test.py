import os
from modules.nl2sqlNet import NL2SQL
from utils.config import init_logger, timer, model_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":
    config = model_config()
    with timer('initializing'):
        nl2sql = NL2SQL(config, epochs=4, batch_size=16, step_batch_size=16, max_len=128, debug=True)
    # with timer('训练'):
    #     # nl2sql.train(batch_size=16, step_batch_size=16)
    #     nl2sql.train()
    with timer('验证'):
        nl2sql.test(batch_size=64, step_batch_size=8, do_evaluate=True, do_test=False)
    # with timer('predicting'):
    #     nl2sql.test(batch_size=32, step_batch_size=16, do_evaluate=False, do_test=True)
