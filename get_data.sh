#!/bin/sh
WORKDIR=./star_trek/data/
mkdir -p ${WORKDIR}
wget -P $WORKDIR https://github.com/alexcg1/startrek-character-lines/raw/master/startrek_tng.csv 
