#!/bin/bash

declare -a targets=("aa2ar" "aces" "andr" "cah2" "cdk2" "dpp4" "dyr" "egfr"
                    "esr1" "esr2" "fa10" "fnta" "hivpr" "hivrt" "mk14" "mmp13"
                    "parp1" "pgh2" "ppara" "pparg" "prgr" "src" "thrb" "vgfr2")

for t in "${targets[@]}"
do
    wget http://dude.docking.org/targets/$t/$t.tar.gz
    tar -xvzf $t.tar.gz
    rm $t.tar.gz
    rm $t/*.gz
done

