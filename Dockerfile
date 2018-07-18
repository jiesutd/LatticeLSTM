FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common
RUN apt-get install curl

RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install build-essential python2.7 python2.7-dev -y

RUN curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | python2.7

RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip2.7 1

RUN mkdir -p /app  
WORKDIR /app


RUN pip install tornado
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl

COPY . /app

EXPOSE 5002

ENTRYPOINT ["sh","/app/server.sh"]
CMD [""]