#!/bin/sh
task_name=$1
cd ${task_name}
python train.py --train_lawithea true