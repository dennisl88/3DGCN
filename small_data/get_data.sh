#!/bin/bash

declare -a targets=("ada" "ampc" "comt" "cxcr4" "fabp4" "fak1" "fkb1a" "fpps"
                    "glcm" "grik1" "hivint" "hs90a" "hxk4" "inha" "kith" "mapk2"
                    "mk01" "mk10" "nram" "pa2ga" "pur2" "pygm" "sahh" "xiap")

for t in "${targets[@]}"
do
    wget http://dude.docking.org/targets/$t/$t.tar.gz
    tar -xvzf $t.tar.gz
    rm $t.tar.gz
    rm $t/*.gz
done

