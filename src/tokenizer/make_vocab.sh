#!/bin/sh

## setup
dpath=../../data/raw
tpath=../../data/tokenizer
vocab_size=8000

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 vocab_trainer.py --data $dpath/all_data.txt --size $vocab_size --output $tpath/vocab_$vocab_size