# Assignment 1 
## Question 1
### Overview
- POSIX Shell Implementation in C++

### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```shell
g++ main.cpp; ./a.out
```

### Implementations and Assumptions
- Shell prompt : full path is displayed instead of ~
- there's problem with cd - . Other type of commands, pwd and echo are working properly
- ls is working in all given cases:
    ● ls
    ● ls -a
    ● ls -l
    ● ls .
    ● ls ..
    ● ls ~
    ● ls -a -l
    ● ls -la / ls -al
    ● ls <Directory/File_name>
    ● ls -<flags> <Directory/File_name>

- implemented fg and bg processes using execvp. When & is used, pid will be printed on console.
- pinfo without popen() is implemented. displays pid , Process Status , memory , and Executable Path
- recursive search is implemented
- piping is implemented partially (redirection problems)
- history and history <number> are implemented
- signals are implemented



