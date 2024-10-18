#!/bin/bash

# [‘#’, ‘$’, ‘*’, ‘@’]
# declare -a fourspecsymbs=("#" "$" "*" "@")

while IFS= read -r sentence || [ -n "$sentence" ] 
do
    declare -a engwords=()
    declare -a specsymb=()
    declare -a revengwords=()

    read -ra testwords <<< "$sentence"
    # echo "${sentence}"
    # echo "${testwords}" "${testwords[*]}"
    # wordsnumber=${#testwords[@]}
    # echo "${wordsnumber}"

    for word in "${testwords[@]}" 
    do
        if [[ "$word" = @("#"|"$"|"*"|"@") ]]
        then
            specsymb+=("$word")
        else
            engwords+=("$word")
        fi
    done

    # specsymbnumber=${#specsymb[@]}
    engwordsnumber=${#engwords[@]}

    for (( i = engwordsnumber - 1; i >= 0; i-- )) 
    do
        revengwords+=("${engwords[i]}")
    done

    revsentence=""
    wordindex=0
    spesymbindex=0

    for word in "${testwords[@]}" 
    do
        revsentence+=" "
        if [[ "$word" = @("#"|"$"|"*"|"@") ]]
        then
            revsentence+="${specsymb[spesymbindex]}"
            ((spesymbindex++))
        else
            revsentence+="${revengwords[wordindex]}"
            ((wordindex++))
        fi
    done

    echo "${revsentence:1}"
done < "$1"

