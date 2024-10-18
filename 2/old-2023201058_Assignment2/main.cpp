#include "headers.h"
#include "prompt.cpp"
#include "cd.cpp"
#include "pwd.cpp"
#include "echo.cpp"
#include "ls.cpp"
#include "fg.cpp"
#include "bg.cpp"
#include "pinfo.cpp"
#include "search.cpp"
#include "ioredirection.cpp"
#include "pipecom.cpp"
#include "signals.cpp"
// #include "autocom.cpp"
using namespace std;
vector<pid_t> backgroundProcesses; // Vector to store background process PIDs


int main() {

    // Register signal handlers
    registerSignalHandlers();

    while (true) {
        displayPrompt();

        string input;
        getline(cin, input);


        // Check for EOF (CTRL-D)
        if (cin.eof()) {
            cout << "CTRL-D pressed. Exiting shell." << endl;
            break;
        }


        if (input.empty()) {
            continue;
        }
        

        vector<string> commands;
        size_t pos = 0;

        // Split input by semicolon
        while ((pos = input.find(';')) != string::npos) {
            commands.push_back(input.substr(0, pos));
            input.erase(0, pos + 1);
        }
        commands.push_back(input);

        int pipeIndex = 0; // Initialize pipe index

        for (const string& command : commands) {
            vector<string> tokens = tokenizeInput(command);

            // Check if the last token is '&'
            bool isBackground = false;
            if (!tokens.empty() && tokens.back() == "&") {
                isBackground = true;
                tokens.pop_back();
            }

            // Create pipes for inter-process communication
            int pipeCount = count(tokens.begin(), tokens.end(), "|");
            vector<int> pipes(pipeCount * 2, -1);
            for (int i = 0; i < pipeCount; ++i) {
                pipe(&pipes[i * 2]);
            }

            // Handle I/O redirection
            handleRedirection(tokens);

            // Handle built-in commands
            if (!tokens.empty()) {
                if (tokens[0] == "cd") {
                    changeDirectory(tokens);
                } else if (tokens[0] == "echo") {
                    echoCommand(tokens);
                } else if (tokens[0] == "pwd") {
                    printWorkingDirectory();
                } else if (tokens[0] == "ls") {
                    listDirectory(tokens);
                } else if (tokens[0] == "pinfo") {
                    if (tokens.size() > 1) {
                        processInfo(tokens[1]);
                    } else {
                        processInfo(to_string(getpid()));
                    }
                } else if (tokens[0] == "search") {
                    if (tokens.size() > 1) {
                        bool result = searchFileOrFolder(tokens[1], ".");
                        cout << (result ? "True" : "False") << endl;
                    } else {
                        cout << "Usage: search <file_or_folder_name>" << endl;
                    }
                } else {
                    if (isBackground) {
                        // Execute the command in the background
                        executeInBackground(tokens, pipes, pipeIndex);
                    } else {
                        // Execute the command in the foreground
                        executeInForeground(tokens, pipes, pipeIndex);
                    }
                    pipeIndex++; // Increment the pipe index
                }
            }
        }
    }

    return 0;
}
