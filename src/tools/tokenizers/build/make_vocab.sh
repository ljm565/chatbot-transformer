#!/bin/sh

## setup
dpath=${1}
tpath=${2}
vocab_size=${3}

echo "Data path: $dpath"
echo "Tokenizers path: $tpath"
echo "Vocab size: $vocab_size"

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 ${4} --data $dpath/all_data.txt --size $vocab_size --output $tpath/vocab_$vocab_size