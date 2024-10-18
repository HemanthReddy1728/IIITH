#include "headers.h"
#include <unistd.h>
using namespace std;

// Function to reset stdin and stdout to their original file descriptors
// void reset_io(int saved_stdin, int saved_stdout) {
//     dup2(saved_stdin, STDIN_FILENO);
//     dup2(saved_stdout, STDOUT_FILENO);
// }

void segfault_handler(int signal) {
    // std::cerr << "Segmentation fault occurred. Continuing execution." << std::endl;
}

int execute_with_redirection(char* input_file, char* output_file, char* arguments[], bool append_redirection) {
    int saved_stdin = dup(STDIN_FILENO);
    int saved_stdout = dup(STDOUT_FILENO);

    if (strlen(input_file) > 0) {
        int input_fd = open(input_file, O_RDONLY);
        if (input_fd < 0) {
            perror("Input redirection");
            return 1;
        }
        dup2(input_fd, STDIN_FILENO);
        close(input_fd);
    }

    // std::signal(SIGSEGV, segfault_handler);
    int output_fd;
    if (strlen(output_file) > 0) {
        if (append_redirection) { // Open the file in append mode if append redirection is enabled
            output_fd = open(output_file, O_WRONLY | O_CREAT | O_APPEND, 0777);
        } else {
            output_fd = open(output_file, O_WRONLY | O_CREAT | O_TRUNC, 0777);
        }
        if (output_fd < 0) {
            perror("Output redirection");
            return 1;
        }
        // cout << "above dup2" << endl;
        dup2(output_fd, STDOUT_FILENO);
        close(output_fd);
        
            // cout << "below dup2" << endl;
    //     dup2(saved_stdin, STDIN_FILENO);
    // dup2(saved_stdout, STDOUT_FILENO);

    }

    cout << arguments[0] << arguments[1] << arguments[2] << endl;


    if (execvp(arguments[0], arguments) == -1) {
        perror("execvp");
        return 1;
    }
    dup2(saved_stdin, STDIN_FILENO);
    dup2(saved_stdout, STDOUT_FILENO);
    // Reset stdin and stdout to their original file descriptors
    cout << "below dup2 ret 0" << endl;
    return 0;
}


/*
int execute_with_redirection(char* input_file, char* output_file, char* arguments[], bool append_redirection) {
    int saved_stdin = dup(STDIN_FILENO);
    int saved_stdout = dup(STDOUT_FILENO);
    // cout << saved_stdin << " " << saved_stdout << endl;
    int input_fd = -1;
    int output_fd = -1;

    if (strlen(input_file) > 0) {
        input_fd = open(input_file, O_RDONLY);
        if (input_fd < 0) {
            perror("Input redirection");
            return 1;
        }
    }

    if (strlen(output_file) > 0) {
        if (append_redirection) {
            output_fd = open(output_file, O_WRONLY | O_CREAT | O_APPEND, 0644);
        } else {
            output_fd = open(output_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        }

        if (output_fd < 0) {
            perror("Output redirection");
            return 1;
        }
    }
    if (execvp(arguments[0], arguments) == -1) {
        perror("execvp");
        return 1;
    }
    // cout << input_fd << " " << output_fd << endl;
    if (input_fd >= 0) {
        dup2(input_fd, STDIN_FILENO);
        close(input_fd);
    }
    // cout << input_fd << " " << output_fd << endl;
    if (output_fd >= 0) {
        dup2(output_fd, STDOUT_FILENO);
        close(output_fd);
    }
    // cout << input_fd << " " << output_fd << endl;
    // cout << input_fd << " " << output_fd << endl;

    // Reset stdin and stdout to their original file descriptors
    // dup2(saved_stdin, STDIN_FILENO);
    // dup2(saved_stdout, STDOUT_FILENO);
    return 0;
}*/