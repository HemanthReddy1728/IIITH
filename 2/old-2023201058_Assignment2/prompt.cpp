
// // Trim whitespace from both ends of a string
// string trim(const string &str) {
//     auto start = str.begin();
//     while (start != str.end() && isspace(*start)) {
//         ++start;
//     }

//     auto end = str.end();
//     do {
//         --end;
//     } while (distance(start, end) > 0 && isspace(*end));

//     return {start, end + 1};
// }

// void prompt() {
//     char hostname[128];
//     gethostname(hostname, sizeof(hostname));

//     char *cwd = getcwd(NULL, 0);
//     char *home = getenv("HOME");

//     if (cwd && home && strncmp(cwd, home, strlen(home)) == 0) {
//         cout << getlogin() << "@" << hostname << ":~" << (cwd + strlen(home)) << "> ";
//         // cout << getlogin() << "@" << hostname << ":~" << (cwd) << "> ";
//     } else {
//         cout << getlogin() << "@" << hostname << ":" << (cwd ? cwd : "") << "> ";
//     }

//     free(cwd);
// }

// void execute_command(const string &command, int background) {
//     pid_t pid = fork();

//     if (pid == 0) {
//         // Child process
//         // Use the system function to execute the command
//         system(command.c_str());
//         exit(0);
//     } else if (pid > 0) {
//         // Parent process
//         if (!background) {
//             // Wait for the child to complete if not a background process
//             waitpid(pid, NULL, 0);
//         } else {
//             cout << "Background process with PID: " << pid << endl;
//         }
//     } else {
//         perror("fork");
//     }
// }

// int main() {
//     char input[MAX_INPUT_SIZE];

//     while (true) {
//         prompt();

//         if (!cin.getline(input, MAX_INPUT_SIZE)) {
//             break;
//         }

//         int background = 0;

//         // Check if the last character is '&'
//         if (input[strlen(input) - 1] == '&') {
//             background = 1;
//             input[strlen(input) - 1] = '\0';  // Remove '&' and newline
//         }

//         // Check if the input is "exit" command
//         if (strcmp(input, "exit") == 0) {
//             break;  // Exit loop on "exit" command
//         }
        
//         // Tokenize input by ';'
//         string command_string(input);
//         size_t start = 0;
//         size_t end = command_string.find(';');

//         while (end != string::npos) {
//             string command = trim(command_string.substr(start, end - start));
//             if (!command.empty()) {
//                 execute_command(command, background);
//             }
//             start = end + 1;
//             end = command_string.find(';', start);
//         }

//         string last_command = trim(command_string.substr(start));
//         if (!last_command.empty()) {
//             execute_command(last_command, background);
//         }
//     }

//     return 0;
// }


// display_prompt.cpp

// #include "display_prompt.h"

// void display_prompt() {
//     char* user = getenv("USER");
//     char hostname[1024];
//     gethostname(hostname, sizeof(hostname));
//     char cwd[1024];
//     getcwd(cwd, sizeof(cwd));
//     std::cout << user << "@" << hostname << ":" << cwd << "> ";
// }

#include "headers.h"
using namespace std;
void displayPrompt() {
// void displayPrompt(bool printPrompt = true) {
//     if (printPrompt) {
        char* username = getenv("USER");
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        char cwd[1024];
        getcwd(cwd, sizeof(cwd));

        // Replace the home directory with "~"
        string homeDir = getenv("HOME");
        string currentDir(cwd);

        if (currentDir.find(homeDir) == 0) {
            currentDir.replace(0, homeDir.length(), "~");
        }

        cout << username << "@" << hostname << ":" << currentDir << "> ";
    // }
}
