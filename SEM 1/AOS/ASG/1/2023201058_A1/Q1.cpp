#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// Assignment1_1/1

int main(int argc, char *argv[])
{
    mkdir("./Assignment1_1", 0777);
    // // if file does not have in directory
    // // then file foo.txt is created.
    // int fd = open("foo.txt", O_RDONLY | O_CREAT);
  
    // printf("fd = %d\n", fd);
  
    // if (fd == -1) {
    //     // print which type of error have in a code
    //     printf("Error Number % d\n", errno);
  
    //     // print program detail "Success or failure"
    //     perror("Program");
    // }
    // return 0;
}