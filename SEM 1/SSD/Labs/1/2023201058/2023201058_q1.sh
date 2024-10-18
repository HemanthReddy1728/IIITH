#!/bin/bash

sort $1 > sortedfile
cat sortedfile | head -$((($(cat sortedfile | wc -l)+1)/2)) | tail -1
rm sortedfile
