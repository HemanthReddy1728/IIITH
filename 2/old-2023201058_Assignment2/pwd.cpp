#include "headers.h"
using namespace std;

void printWorkingDirectory() {
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    cout << cwd << endl;
}
