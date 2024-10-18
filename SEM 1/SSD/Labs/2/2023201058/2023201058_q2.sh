#!/bin/bash

touch permlist && rm permlist
ls -l "$1" > permlist

grep "rwx-[w-]-[r-][w-]x" permlist

rm permlist

# grep "rwx-[w-]-[r-][w-]x" "$(ls -l "$1")"
# cat "$(ls -l "$1")" | xargs grep "rwx-[w-]-[r-][w-]x"