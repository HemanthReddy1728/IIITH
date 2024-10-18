#include "headers.h"
using namespace std;

void changeDirectory(const vector<string>& args) {
    if (args.size() > 2) {
        cerr << "Invalid arguments for cd" << endl;
        return;
    }

    string targetDir = (args.size() == 1) ? getenv("HOME") : args[1];

    if (targetDir == "~") {
        targetDir = getenv("HOME");
    } else if (targetDir == "-") {
        char* oldDir = getenv("OLDPWD");
        if (oldDir == nullptr) {
            cerr << "OLDPWD not set" << endl;
            return;
        }
        targetDir = oldDir;
    }

    int result = chdir(targetDir.c_str());
    if (result != 0) {
        cerr << "Error changing directory: " << strerror(errno) << endl;
    } else {
        setenv("OLDPWD", getenv("PWD"), 1);
    }
}
