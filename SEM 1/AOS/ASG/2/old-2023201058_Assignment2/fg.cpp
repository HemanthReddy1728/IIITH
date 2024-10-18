#include "headers.h"
using namespace std;

// void executeInForeground(const vector<string>& args) {
//     // Convert args to char* array
//     const char** argv = new const char*[args.size() + 1];
//     for (size_t i = 0; i < args.size(); ++i) {
//         argv[i] = args[i].c_str();
//     }
//     argv[args.size()] = nullptr;

//     int pid = fork();
//     if (pid == 0) {
//         // Child process
//         execvp(argv[0], const_cast<char* const*>(argv));
//         cerr << "Command not found: " << argv[0] << endl;
//         exit(EXIT_FAILURE);
//     } else if (pid > 0) {
//         // Parent process
//         int status;
//         waitpid(pid, &status, 0);
//     } else {
//         cerr << "Fork failed." << endl;
//     }

//     delete[] argv;
// }

void executeInForeground(vector<string>& tokens, vector<int>& pipes, int pipeIndex) {
    // Convert vector of strings to a char* array for execvp
    vector<char*> args;
    for (string& token : tokens) {
        args.push_back(&token[0]);
    }
    args.push_back(nullptr); // Null-terminate the argument list

    // Create a child process
    pid_t childPid = fork();

    if (childPid == 0) {
        // Child process
        if (pipeIndex > 0) {
            // Redirect input from the previous pipe
            dup2(pipes[(pipeIndex - 1) * 2], STDIN_FILENO);
        }

        // Redirect output to the next pipe or file if using > or >>
        if (pipeIndex < pipes.size() / 2) {
            dup2(pipes[pipeIndex * 2 + 1], STDOUT_FILENO);
        }

        // Close pipe file descriptors
        for (int i = 0; i < pipes.size(); ++i) {
            close(pipes[i]);
        }

        // Execute the command in the foreground
        if (execvp(args[0], args.data()) == -1) {
            cerr << "Command execution failed" << endl;
            exit(1);
        }
    } else if (childPid < 0) {
        cerr << "Fork failed." << endl;
    } else {
        // Parent process
        // Wait for the child process to complete
        int status;
        waitpid(childPid, &status, 0);

        if (WIFEXITED(status)) {
            // Child process exited normally
            int exitStatus = WEXITSTATUS(status);
            cout << "Child process exited with status: " << exitStatus << endl;
        } else if (WIFSIGNALED(status)) {
            // Child process terminated due to a signal
            int signalNumber = WTERMSIG(status);
            cout << "Child process terminated with signal: " << signalNumber << endl;
        }
    }
}
