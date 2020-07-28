#!/bin/sh
WORKDIR=./south_park/data/
mkdir -p ${WORKDIR}
git clone https://github.com/alexcg1/startrek-character-lines.git $WORKDIR
python prepare_data.py
