#include "headers.h"
using namespace std;

volatile sig_atomic_t foregroundProcessPid = 0;

void ctrlZHandler(int signum) {
    if (foregroundProcessPid != 0) {
        kill(foregroundProcessPid, SIGTSTP);
        std::cout << std::endl;
        foregroundProcessPid = 0;
    }
}

void ctrlCHandler(int signum) {
    if (foregroundProcessPid != 0) {
        kill(foregroundProcessPid, SIGINT);
        std::cout << std::endl;
    }
}

void registerSignalHandlers() {
    signal(SIGTSTP, ctrlZHandler);
    signal(SIGINT, ctrlCHandler);
}
