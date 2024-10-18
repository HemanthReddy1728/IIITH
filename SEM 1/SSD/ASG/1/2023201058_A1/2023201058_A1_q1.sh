#!/bin/bash

direction_determiner()
{

    if [[ "$(bc <<< "$1 < 0")" == "1" && "$(bc <<< "$2 < 0")" == "1" ]]
    then
        echo "SW"
    elif [[ "$(bc <<< "$1 == 0")" == "1" && "$(bc <<< "$2 < 0")" == "1" ]]
    then
        echo "S"
    elif [[ "$(bc <<< "$1 < 0")" == "1" && "$(bc <<< "$2 == 0")" == "1" ]]
    then
        echo "W"
    elif [[ "$(bc <<< "$1 < 0")" == "1" && "$(bc <<< "$2 > 0")" == "1" ]]
    then
        echo "NW"
    elif [[ "$(bc <<< "$1 > 0")" == "1" && "$(bc <<< "$2 < 0")" == "1" ]]
    then
        echo "SE"
    elif [[ "$(bc <<< "$1 == 0")" == "1" && "$(bc <<< "$2 > 0")" == "1" ]]
    then
        echo "N"
    elif [[ "$(bc <<< "$1 > 0")" == "1" && "$(bc <<< "$2 == 0")" == "1" ]]
    then
        echo "E"
    elif [[ "$(bc <<< "$1 > 0")" == "1" && "$(bc <<< "$2 > 0")" == "1" ]]
    then
        echo "NE"
    fi

}

read -rp "Jaggu X = " Jaggu_X
# read -rp ", Jaggu Y = " Jaggu_Y
read -rp "Jaggu Y = " Jaggu_Y
read -rp "Police X = " Police_X
# read -rp ", Police Y = " Police_Y
read -rp "Police Y = " Police_Y
read -rp "H = " H

X_difference=$(echo "$Jaggu_X - $Police_X" | bc)
Y_difference=$(echo "$Jaggu_Y - $Police_Y" | bc)

# distance_accu=$(echo "scale=4;sqrt($(echo "$X_difference^2" | bc) + $(echo "$Y_difference^2" | bc))" | bc)
distance_approx=$(printf "%.2f" "$(echo "scale=4;sqrt($(echo "$X_difference^2" | bc) + $(echo "$Y_difference^2" | bc))" | bc)")

while [[ "$H" -gt 0 ]]
do
    read -rp "Police X = " Police_X
    # read -rp ", Jaggu Y = " Jaggu_Y
    read -rp "Police Y = " Police_Y
    # read -rp ", Police Y = " Police_Y
    X_difference=$(echo "$Jaggu_X - $Police_X" | bc)
    Y_difference=$(echo "$Jaggu_Y - $Police_Y" | bc)

    # distance_accu=$(echo "scale=4;sqrt($(echo "$X_difference^2" | bc) + $(echo "$Y_difference^2" | bc))" | bc)
    distance_approx=$(printf "%.2f" "$(echo "scale=4;sqrt($(echo "$X_difference^2" | bc) + $(echo "$Y_difference^2" | bc))" | bc)")

    if [[ "$(bc <<< "$distance_approx < 2")" == "1" ]]
    then
        echo "Location reached"
        break
    fi

    ((H--))
    
    if [[ "$(bc <<< "$distance_approx < 2")" == "0" && "$H" -eq 0 ]]
    then
        echo "Time over"
        break
    fi

    echo "$distance_approx" "$(direction_determiner "$X_difference" "$Y_difference")"

done


