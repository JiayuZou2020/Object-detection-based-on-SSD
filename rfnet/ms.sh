#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

root=$2
date=$3
model=$4
resume=""$root"/"$date"/model/"$model""

python -utt ms.py --data=view --resume=$resume
python -utt ms.py --data=illu --resume=$resume
python -utt ms.py --data=ef --resume=$resume


#ubuntu环境中，$表示普通用户，#表示超级管理员