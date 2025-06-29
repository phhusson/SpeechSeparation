#!/bin/bash

# dnr_v3: https://github.com/kwatcharasupat/divide-and-remaster-v3
mkdir -p datasets/dnr_v3
for i in {0..9};do
    wget -c https://zenodo.org/records/12713709/files/dnr-v3-multi-audio.train.tar.gz.0${i}?download=1 -O datasets/dnr_v3/file${i}.tar.gz
done

# VocalSound: https://github.com/YuanGongND/vocalsound
mkdir -p datasets/VocalSound
wget -c https://www.dropbox.com/s/ybgaprezl8ubcce/vs_release_44k.zip?dl=1 -O datasets/VocalSound/file.zip
