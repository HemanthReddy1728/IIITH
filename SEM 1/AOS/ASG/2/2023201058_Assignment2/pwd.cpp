#include "headers.h"
using namespace std;

// Function to print the current directory (pwd command)
void pwd_command() {
    char current_dir[1024];
    if (getcwd(current_dir, sizeof(current_dir)) != nullptr) {
        cout << current_dir << endl;
    } else {
        perror("pwd");
    }
}
