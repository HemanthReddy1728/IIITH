#!/bin/bash

ls $1 | grep -i "^F" | grep -v ".cpp$"