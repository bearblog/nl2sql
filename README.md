# 环境
- python 3.6
- pytorch 1.0.0   
- cuda 10
- 其他python依赖可以参考requirments.txt

# 文件说明
- data/        存放训练和测试数据
- model/       下放了两个哈工大的预训练模型，两个比赛模型，online的是线上提交的模型，另一个是今早上（08-09）重新跑的模型， 
- submit/      存放了对应的预测文件

# 运行
- 切到 code/ 文件夹下 
- 训练： python train.py   
-  预测： python test.py 

- 两个代码都已经在本地调试通过，具体可以查看对应log日志（./code/log_bad_cases/nl2sql.log）

# 复赛docker构建和提交流程

### 将代码打包为镜像

```bash
docker build registry.cn-shenzhen.aliyuncs.com/[命名空间]/[仓库名]:[镜像版本号]
```
例子
```bash
docker build -t registry.cn-shenzhen.aliyuncs.com/nl2sql_bupt/nl2sql:v1.5 .
```

### 测试代码

```bash
nvidia-docker run -v /path/to/test/:/tcdata --name container_name registry.cn-shenzhen.aliyuncs.com/[命名空间]/[仓库名]:[镜像版本号] sh run.sh
```

例子(目前上线版本)
```bash
nvidia-docker run --rm -v /home1/sjb2018/workspace/NL2SQL/test/:/tcdata registry.cn-shenzhen.aliyuncs.com/nl2sql_bupt/nl2sql:v1.5 sh run.sh
```

### 上传镜像

```bash
docker login registry.cn-shenzhen.aliyuncs.com
docker push registry.cn-shenzhen.aliyuncs.com/[命名空间]/[仓库名]:[镜像版本号]
```

