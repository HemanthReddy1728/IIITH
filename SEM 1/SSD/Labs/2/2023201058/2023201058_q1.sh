#!/bin/bash

read -r m
read -ra n

for s in "${n[@]}"
do
    if [ $(("$s" % 2)) -eq 0 ] 
    then
        s=0
    fi

    for (( i=1;i<="$s";i+=2 ))
    do
        for (( j=s;j>=i;j-- ))
        do
            echo -n " "
        done
        
        for (( k=1;k<=i;k++ ))
        do
            echo -n "* "
            sum+=1
        done
        echo ""
    done

    ((--s))
    
    for (( i="$s"-1;i>=1;i-=2 ))
    do
        for (( j=i;j<="$s";j++ ))
        do
            if [ $j -eq "$s" ]
            then
            echo -n " "
            fi
            echo -n " "
        done

        for (( k=1;k<=i;k++ ))
        do
            echo -n "* "
            sum+=1
        done

        echo ""
    done
done