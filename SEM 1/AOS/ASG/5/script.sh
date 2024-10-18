#!/bin/bash

# Function to run commands in sint3
run_sint3_command() {
    sint3 "$@" <<EOF
python3 2023201058_q1.py
exit
EOF
}

# Run commands for sint3 without arguments
run_sint3_command
wait
# # 2 cores
# sint3 -c 2 <<EOF
# python3 2023201058_q1.py;exit
# EOF

# sleep 10

# # 4 cores
# sint3 -c 4 <<EOF
# python3 2023201058_q1.py;exit
# EOF

# sint3 -c 6 <<EOF
# # 6 cores
# python3 2023201058_q1.py
# exit
# EOF

# # 6 cores
# sint3 <<EOF
# python3 2023201058_q1.py;exit;
# EOF

# # 24 cores
# python3 2023201058_q1.py;


# sint3 -c 2 <<EOF
# # 2 cores
# python3 2023201058_q2.py
# exit
# EOF

# sint3 -c 4 <<EOF
# # 4 cores
# python3 2023201058_q2.py
# exit
# EOF

# sint3 -c 6 <<EOF
# # 6 cores
# python3 2023201058_q2.py
# exit
# EOF

# sint3 <<EOF
# # 6 cores
# python3 2023201058_q2.py
# exit
# EOF

# # 24 cores
# python3 2023201058_q2.py


# sint3 -c 2 <<EOF
# # 2 cores
# python3 2023201058_q3.py
# exit
# EOF

# sint3 -c 4 <<EOF
# # 4 cores
# python3 2023201058_q3.py
# exit
# EOF

# sint3 -c 6 <<EOF
# # 6 cores
# python3 2023201058_q3.py
# exit
# EOF

# sint3 <<EOF
# # 6 cores
# python3 2023201058_q3.py
# exit
# EOF

# # 24 cores
# python3 2023201058_q3.py

