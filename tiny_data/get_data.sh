#!/bin/bash

declare -a targets=("ada" "ampc" "comt" "cxcr4" "fabp4" "fak1" "fkb1a" "fpps")

for t in "${targets[@]}"
do
    wget http://dude.docking.org/targets/$t/$t.tar.gz
    tar -xvzf $t.tar.gz
    rm $t.tar.gz
    rm $t/*.gz
done

