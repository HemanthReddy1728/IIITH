#!/bin/bash

max()
{
    if [[ "$1" -gt "$2" ]]
    then
        echo "$1"
    else
        echo "$2"
    fi
}

min()
{
    if [[ "$1" -lt "$2" ]]
    then
        echo "$1"
    else
        echo "$2"
    fi
}

read -rp "N=" N
read -rp "prices = " -a prices

# for (( i = 1 ; i <= N ; i++ ))
# do
#     read -rp "Price on Day $i = " line
#     prices=("${prices[@]}" "$line")
# done

minPrice=${prices[0]}
maxPro=0

for (( i = 0 ; i < N ; i++ ))
do  
    minPrice="$(min "$(echo "$minPrice" | bc)" "$(echo "${prices[$i]}" | bc)")"
    maxPro="$(max "$(echo "$maxPro" | bc)" "$(echo "${prices[$i]} - $minPrice" | bc)")"
done

echo "Maximum Profit:" "$maxPro"
