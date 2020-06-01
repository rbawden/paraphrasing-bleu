#!/bin/bash

#this script extract laser embeddings (see more information: https://github.com/facebookresearch/LASER/tree/master/tasks/embed)
#usage: bash extract_laser_embeds.sh name_of_file (for example bash extract_laser_embeds.sh test.en)
#please add paths to the data and LASER

#input will be automatically tokenized and BPE will be applied
#output: the embeddings are stored in float32 matrices in raw binary format

file_name=$1
DATA=PATH_TO_DATA
LASER=PATH_TO_LASER

#extract language from the given file
lang=${file_name: -2}
file_name_wo_lang=${file_name:: -3}


#get LASER embeddings
bash ${LASER}/tasks/embed/embed.sh ${DATA}/${file_name} $lang ${DATA}/${file_name_wo_lang}.emb.raw.${lang}.laser
