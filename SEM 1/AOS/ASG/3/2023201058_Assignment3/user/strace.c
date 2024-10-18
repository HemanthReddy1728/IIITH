#include "kernel/types.h"
#include "kernel/stat.h"
#include "user.h"

// int
// main(void) {
//     // printf(1, "return val of system call is %d\n", hello());
//     // printf(1, "Congrats !! You have successfully added new system call in xv6 OS :) \n");
//     // exit();
//     printf("Return val of system call is %d\n", hello());
//     printf("Congrats !! You have successfully added new system call in xv6 OS :) \n");
//     exit(1);
// }

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(2, "usage: strace mask command [args]\n");
        exit(1);
    }

    trace(atoi(argv[1]));

    char *execargs[argc - 1];
    for (int i = 0; i < argc - 2; i++)
    {
        execargs[i] = argv[i + 2];
    }
    execargs[argc - 2] = 0;
    exec(execargs[0], execargs);
    fprintf(2, "exec %s failed\n", execargs[0]);
    exit(0);
}