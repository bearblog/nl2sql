# 环境：
## python 3.7
## pytorch 1.1.0   
## cuda 10
## 其他python依赖可以参考requirments.txt

# 文件说明：
## model/       下放了两个哈工大的预训练模型，两个比赛模型，online的是线上提交的模型，另一个是今早上（08-09）重新跑的模型， 
## submit/      存放了对应的预测文件

# 运行：
## 切到 code/ 文件夹下 
## 训练： python train.py   
## 预测： python test.py 

## 两个代码都已经在本地调试通过，具体可以查看对应log日志（./code/log_bad_cases/nl2sql.log）

两个文件基本无区别，只是不是太想写命令行参数..（test应该会高于线上0806的成绩，因为当时提交的时候模型没收敛）
