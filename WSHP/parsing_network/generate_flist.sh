#!/bin/bash
# get all filename in specified path
path=../../data/hico/images/test2015/    # for hico-test
# path=../../data/hico/images/train2015/ # for hico-train
# path=../../data/vcoco/images/test/     # for vcoco-test
# path=../../data/vcoco/images/trainval/ # for vcoco-train
files=$(ls $path)

for filename in $files
do
 echo $filename >> filename.txt
done
