#include "headers.h"
using namespace std;

// Function to handle input and output redirection
void handleRedirection(vector<string>& tokens) {
    int inputFd = -1;
    int outputFd = -1;
    bool appendMode = false;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "<" && i + 1 < tokens.size()) {
            // Input redirection
            inputFd = open(tokens[i + 1].c_str(), O_RDONLY);
            if (inputFd == -1) {
                cerr << "Error opening input file: " << tokens[i + 1] << endl;
                exit(1); // Exit the child process
            }
            dup2(inputFd, STDIN_FILENO);
            close(inputFd);
            tokens.erase(tokens.begin() + i, tokens.begin() + i + 2); // Remove "<" and the filename
        } else if (tokens[i] == ">" && i + 1 < tokens.size()) {
            // Output redirection (overwrite)
            outputFd = open(tokens[i + 1].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (outputFd == -1) {
                cerr << "Error opening output file: " << tokens[i + 1] << endl;
                exit(1); // Exit the child process
            }
            dup2(outputFd, STDOUT_FILENO);
            close(outputFd);
            tokens.erase(tokens.begin() + i, tokens.begin() + i + 2); // Remove ">" and the filename
        } else if (tokens[i] == ">>" && i + 1 < tokens.size()) {
            // Output redirection (append)
            outputFd = open(tokens[i + 1].c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (outputFd == -1) {
                cerr << "Error opening output file: " << tokens[i + 1] << endl;
                exit(1); // Exit the child process
            }
            dup2(outputFd, STDOUT_FILENO);
            close(outputFd);
            appendMode = true;
            tokens.erase(tokens.begin() + i, tokens.begin() + i + 2); // Remove ">>" and the filename
        }
    }

    if (appendMode) {
        // Clear tokens if in append mode to avoid duplicate arguments
        tokens.clear();
    }
}
