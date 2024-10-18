#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fcntl.h"

int
main(int argc, char ** argv)
{
    if (argc < 3) {
        fprintf(2, "%s: execution failed - insufficient number of arguments\n", argv[0]);
        exit(1);
    }

    int mask = atoi(argv[1]);

    if (mask < 0) {
        fprintf(2, "%s: execution failed - invalid mask provided\n", argv[0]);
        exit(1);
    }

    if (trace(mask) < 0) {
        fprintf(2, "%s: execution failed\n", argv[0]);
        exit(1);
    }

    exec(argv[2], argv + 2);
    exit(0);
}