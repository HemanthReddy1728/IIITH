#include "headers.h"
using namespace std;

void run_command_background(const char* input, vector<pid_t>& bg_processes) {
    // Fork a child process
    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        // Create a new session and detach from the controlling terminal
        if (setsid() == -1) {
            perror("setsid");
            exit(EXIT_FAILURE);
        }

        // Split the input into command and arguments
        char* command = strtok(const_cast<char*>(input), " \t\n");
        if (command == nullptr) {
            cerr << "Invalid command" << endl;
            exit(EXIT_FAILURE);
        }

        // Build argument list for execvp
        char* args[1024]; // Assuming a maximum of 1024 arguments
        args[0] = const_cast<char*>(command);
        int arg_count = 1;

        char* arg = strtok(nullptr, " \t\n");
        while (arg != nullptr) {
            args[arg_count] = arg;
            ++arg_count;
            arg = strtok(nullptr, " \t\n");
        }

        args[arg_count] = nullptr;

        // Execute the command in the background
        if (execvp(command, args) == -1) {
            perror("execvp");
            exit(EXIT_FAILURE);
        }
    } else if (pid > 0) {
        // Parent process
        bg_processes.push_back(pid); // Store the background process PID
        cout << "[" << pid << "]" << endl; // Print the PID of the background process
    } else {
        perror("fork");
    }
}





void run_command_foreground(const char* input) {
    // Fork a child process
    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        // Split the input into command and arguments
        char* command = strtok(const_cast<char*>(input), " \t\n");
        if (command == nullptr) {
            cerr << "Invalid command" << endl;
            exit(EXIT_FAILURE);
        }

        // Build argument list for execvp
        char* args[1024]; // Assuming a maximum of 1024 arguments
        args[0] = command;
        int arg_count = 1;

        char* arg = strtok(nullptr, " \t\n");
        while (arg != nullptr) {
            args[arg_count] = arg;
            ++arg_count;
            arg = strtok(nullptr, " \t\n");
        }

        args[arg_count] = nullptr;

        // Execute the command in the foreground
        if (execvp(command, args) == -1) {
            perror("execvp");
            exit(EXIT_FAILURE);
        }
    } else if (pid > 0) {
        // Parent process
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            // cout << "Foreground process exited with status " << WEXITSTATUS(status) << endl;
        }
    } else {
        perror("fork");
    }
}

void check_background_processes(vector<pid_t>& bg_processes) {
    for (auto it = bg_processes.begin(); it != bg_processes.end();) {
        pid_t pid = *it;
        int status;

        if (waitpid(pid, &status, WNOHANG) > 0) {
            // The background process has exited
            cout << "Background process with PID " << pid << " has exited." << endl;
            it = bg_processes.erase(it);
        } else {
            ++it;
        }
    }
}
