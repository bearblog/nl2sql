FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
ADD . /competition
WORKDIR /competition
RUN pip --no-cache-dir install  -r requirements.txt
CMD ["sh", "run.sh"]