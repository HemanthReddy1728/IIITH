#!/bin/bash

cd "$1" || exit
touch cpplist && rm cpplist
ls *.cpp > cpplist

while IFS= read -r line || [ -n "$line" ]
do
    grep "#include" "$line"
    # grep "#include" "$line" | tr -d '\n'
    # grep "#include" "$line" | tr "'\n' ' '"
    # echo
done < cpplist
# done < "$(ls *.cpp)"

rm cpplist