#include "prompt.cpp"
#include "cd.cpp"
#include "pwd.cpp"
#include "echo.cpp"
#include "ls.cpp"
#include "bgfg.cpp"
#include "pinfo.cpp"
#include "search.cpp"
#include "redirect.cpp"
#include "signals.cpp"
#include "history.cpp"
#include "auto.cpp"
#include "headers.h"
using namespace std;

void sigintHandler(int signum) {
    cout << endl; // Insert a newline character
    display_prompt(); // Display the shell prompt again
}

int main() {
    SignalHandler signal_handler;
    signal_handler.RegisterHandlers();
    signal(SIGINT, sigintHandler);

    History history;
    int currentHistoryIndex = -1;

string partial; // Add this line to declare the partial variable

while (1) {
    display_prompt();
    vector<pid_t> bg_processes;

    int c;
    struct termios old_termios, new_termios;
    tcgetattr(STDIN_FILENO, &old_termios);
    new_termios = old_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    do {
        c = getchar();
    } while (c != EOF && isspace(c)); // Skip leading whitespace characters, including newline
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios);

    if (c == 27) {
        c = getchar();
        if (c == 91) {
            c = getchar();
            if (c == 65) { // UP arrow key pressed
                if (history.isEmpty()) {
                    continue;
                }
                if (currentHistoryIndex == -1) {
                    currentHistoryIndex = history.size() - 1;
                } else if (currentHistoryIndex > 0) {
                    currentHistoryIndex--;
                }
                string previousCommand = history.getCommand(currentHistoryIndex);
                if (!previousCommand.empty()) {
                    std::cout << "\033[A"; // Move cursor up one line
                    std::cout << "\r" << previousCommand; // Clear the line and print the previous command
                }
            }
        }
    }

    char input[1024];
    char *arguments[1024];

    if (c == '\t') {
        // Trigger autocomplete
        if (!partial.empty()) {
            std::vector<std::string> suggestions = autocomplete(partial);
            if (suggestions.empty()) {
                // No matches found
                std::cout << "\nNo matches found.\n";
            } else if (suggestions.size() == 1) {
                // Complete the input with the single match
                partial = suggestions[0];
                strcat(input, partial.c_str()); // Concatenate partial to input
                std::cout << partial;
            } else {
                // Display multiple matches
                std::cout << "\n";
                for (const std::string& suggestion : suggestions) {
                    std::cout << suggestion << " ";
                }
            }
        }
    }


        if (fgets(input, sizeof(input), stdin) == nullptr) {
            echo_command(nullptr);
            break; // Exit the shell if EOF is encountered (e.g., Ctrl+D)
        }

        string inputStr(input);
        history.addCommand(inputStr);

        int len = strlen(input);
        for (int i = 0; i < len; i++) {
            if (input[i] == '\n') {
                input[i] = '\0';
                break;
            }
        }

        if (strcmp(input, "exit") == 0 || strcmp(input, "quit") == 0) {
            break; // Exit the shell if the user enters "exit" or "quit"
        }

        char *pipe_commands[1024];
        char *token;
        token = strtok(input, "|");
        int num_pipes = 0;
        while (token != nullptr) {
            pipe_commands[num_pipes] = token;
            num_pipes++;
            token = strtok(nullptr, "|");
        }

        if (num_pipes > 1) {
            int pipe_fds[num_pipes - 1][2];

            for (int i = 0; i < num_pipes; i++) {
                // if (i < num_pipes - 1) {
                    if (pipe(pipe_fds[i]) == -1) {
                        perror("pipe");
                        exit(EXIT_FAILURE);
                    }
                // }

                char *command = strtok(pipe_commands[i], " \t");
                int no_of_tokens = 0;
                while (command != nullptr) {
                    arguments[no_of_tokens] = command;
                    no_of_tokens++;
                    command = strtok(nullptr, " \t");
                }
                arguments[no_of_tokens] = nullptr;

                if (no_of_tokens == 0)
                    continue;

                char input_file[1024] = "";
                char output_file[1024] = "";
                bool append_redirection = false;
                int j = 0;

                while (arguments[j] != nullptr) {
                    if (strcmp(arguments[j], "<") == 0) {
                        if (arguments[j + 1] != nullptr) {
                            strcpy(input_file, arguments[j + 1]);
                            // Nullify the redirection tokens and their arguments
                            arguments[j] = nullptr;
                            arguments[j + 1] = nullptr;
                        } else {
                            cerr << "Invalid input redirection syntax." << endl;
                            exit(EXIT_FAILURE);
                        }
                    } else if (strcmp(arguments[j], ">") == 0) {
                        if (arguments[j + 1] != nullptr) {
                            strcpy(output_file, arguments[j + 1]);
                            // Nullify the redirection tokens and their arguments
                            arguments[j] = nullptr;
                            arguments[j + 1] = nullptr;
                        } else {
                            cerr << "Invalid output redirection syntax." << endl;
                            exit(EXIT_FAILURE);
                        }
                    } else if (strcmp(arguments[j], ">>") == 0) {
                        if (arguments[j + 1] != nullptr) {
                            strcpy(output_file, arguments[j + 1]);
                            append_redirection = true;
                            // Nullify the redirection tokens and their arguments
                            arguments[j] = nullptr;
                            arguments[j + 1] = nullptr;
                        } else {
                            cerr << "Invalid append redirection syntax." << endl;
                            exit(EXIT_FAILURE);
                        }
                    } else {
                        j++;
                    }
                }

                if (i == 0) {
                    pid_t child_pid = fork();
                    if (child_pid == 0) {
                        // First command in the pipeline
                        if (i < num_pipes - 1) {
                            if (dup2(pipe_fds[i][1], STDOUT_FILENO) == -1) {
                                perror("dup2");
                                exit(EXIT_FAILURE);
                            }
                            close(pipe_fds[i][0]);
                            close(pipe_fds[i][1]);
                        }

                        int exec_status = execute_with_redirection(input_file, output_file, arguments, append_redirection);

                        if (exec_status != 0) {
                            exit(EXIT_FAILURE);
                        }
                    } else if (child_pid > 0) {
                        if (i < num_pipes - 1) {
                            close(pipe_fds[i][1]);
                        }
                        int status;
                        waitpid(child_pid, &status, 0);
                    } else {
                        perror("fork");
                    }
                } else if (i == num_pipes - 1) {
                    pid_t child_pid = fork();
                    if (child_pid == 0) {
                        // Last command in the pipeline
                        if (dup2(pipe_fds[i - 1][0], STDIN_FILENO) == -1) {
                            perror("dup2");
                            exit(EXIT_FAILURE);
                        }
                        close(pipe_fds[i - 1][1]);
                        close(pipe_fds[i - 1][0]);

                        int exec_status = execute_with_redirection(input_file, output_file, arguments, append_redirection);

                        if (exec_status != 0) {
                            exit(EXIT_FAILURE);
                        }
                    } else if (child_pid > 0) {
                        close(pipe_fds[i - 1][0]);
                        int status;
                        waitpid(child_pid, &status, 0);
                    } else {
                        perror("fork");
                    }
                } else {
                    pid_t child_pid = fork();
                    if (child_pid == 0) {
                        // Intermediate commands in the pipeline
                        if (dup2(pipe_fds[i - 1][0], STDIN_FILENO) == -1) {
                            perror("dup2");
                            exit(EXIT_FAILURE);
                        }
                        if (dup2(pipe_fds[i][1], STDOUT_FILENO) == -1) {
                            perror("dup2");
                            exit(EXIT_FAILURE);
                        }
                        close(pipe_fds[i - 1][1]);
                        close(pipe_fds[i - 1][0]);
                        close(pipe_fds[i][1]);
                        close(pipe_fds[i][0]);

                        int exec_status = execute_with_redirection(input_file, output_file, arguments, append_redirection);

                        if (exec_status != 0) {
                            exit(EXIT_FAILURE);
                        }
                    } else if (child_pid > 0) {
                        close(pipe_fds[i - 1][0]);
                        close(pipe_fds[i - 1][1]);
                        int status;
                        waitpid(child_pid, &status, 0);
                    } else {
                        perror("fork");
                    }
                }
            }

        } else {
            char *command = strtok(input, " \t");
            int no_of_tokens = 0;
            int redirection_flag = 0;
            while (command != nullptr) {
                arguments[no_of_tokens] = command;
                if (strcmp(command, "<") == 0 || strcmp(command, ">") == 0 || strcmp(command, ">>") == 0)
                    ++redirection_flag;
                ++no_of_tokens;
                command = strtok(nullptr, " \t");
            }
            arguments[no_of_tokens] = nullptr;

            if (no_of_tokens == 0)
                continue;

            char input_file[1024] = "";
            char output_file[1024] = "";
            bool input_redirection = false, output_redirection = false, append_redirection = false;
            int i = 0;

            while (arguments[i] != nullptr) {
                if (strcmp(arguments[i], "<") == 0) {
                    if (arguments[i + 1] != nullptr) {
                        strcpy(input_file, arguments[i + 1]);
                        input_redirection = true;
                        // Nullify the redirection tokens and their arguments
                        arguments[i] = nullptr;
                        arguments[i + 1] = nullptr;
                    } else {
                        cerr << "Invalid input redirection syntax." << endl;
                        exit(EXIT_FAILURE);
                    }
                } else if (strcmp(arguments[i], ">") == 0) {
                    if (arguments[i + 1] != nullptr) {
                        strcpy(output_file, arguments[i + 1]);
                        output_redirection = true;
                        // Nullify the redirection tokens and their arguments
                        arguments[i] = nullptr;
                        arguments[i + 1] = nullptr;
                    } else {
                        cerr << "Invalid output redirection syntax." << endl;
                        exit(EXIT_FAILURE);
                    }
                } else if (strcmp(arguments[i], ">>") == 0) {
                    if (arguments[i + 1] != nullptr) {
                        strcpy(output_file, arguments[i + 1]);
                        append_redirection = true;
                        // Nullify the redirection tokens and their arguments
                        arguments[i] = nullptr;
                        arguments[i + 1] = nullptr;
                    } else {
                        cerr << "Invalid append redirection syntax." << endl;
                        exit(EXIT_FAILURE);
                    }
                } else {
                    i++;
                }
            }

            // Nullify the remaining arguments after redirection tokens are removed
            if (input_redirection || output_redirection || append_redirection) {
                int j = 0;
                for (int k = 0; arguments[k] != nullptr; k++) {
                    arguments[j] = arguments[k];
                    j++;
                }
                arguments[j] = nullptr;
            }

            if (strcmp(arguments[0], "exit") == 0 || strcmp(arguments[0], "quit") == 0) {
                break; // Exit the shell if the user enters "exit" or "quit"
            } else if (strcmp(arguments[0], "cd") == 0 && redirection_flag == 0) {
                cd_command(arguments[1]);
            } else if (strcmp(arguments[0], "pwd") == 0 && redirection_flag == 0) {
                pwd_command();
            } else if (strcmp(arguments[0], "echo") == 0 && redirection_flag == 0) {
                echo_command(arguments);
            } else if (strcmp(arguments[0], "ls") == 0 && redirection_flag == 0) {
                bool show_hidden = false;
                bool long_format = false;
                const char *path = ".";

                int i = 1;
                while (arguments[i] != nullptr) {
                    if (arguments[i][0] == '-') {
                        for (int j = 1; arguments[i][j] != '\0'; j++) {
                            if (arguments[i][j] == 'a') {
                                show_hidden = true;
                            } else if (arguments[i][j] == 'l') {
                                long_format = true;
                            } else {
                                cerr << "Invalid flag: " << arguments[i][j] << endl;
                                break;
                            }
                        }
                    } else {
                        path = arguments[i];
                    }
                    ++i;
                }
                vector<string> paths;
                paths.push_back(path);
                list_files(paths, show_hidden, long_format);
            } else if (strcmp(arguments[0], "pinfo") == 0) {
                if (no_of_tokens == 1) {
                    pinfo_command();
                } else if (no_of_tokens == 2) {
                    int pid = atoi(arguments[1]);
                    if (pid > 0) {
                        pinfo_command(arguments[1]);
                    } else {
                        cerr << "Invalid PID" << endl;
                    }
                } else {
                    cerr << "Usage: pinfo [pid]" << endl;
                }
            } else if (strcmp(arguments[0], "search") == 0) {
                if (no_of_tokens == 2) {
                    string target = arguments[1];
                    if (search_recursive(".", target)) {
                        cout << "True" << endl;
                    } else {
                        cout << "False" << endl;
                    }
                } else {
                    cerr << "Usage: search <filename or foldername>" << endl;
                }
            } else if (strcmp(arguments[0], "history") == 0) {
                if (no_of_tokens == 1) {
                    vector<string> fullHistory = history.getHistory();
                    for (const string &cmd : fullHistory) {
                        cout << cmd << endl;
                    }
                } else if (no_of_tokens == 2) {
                    int num = atoi(arguments[1]);
                    vector<string> limitedHistory = history.getHistory(num);
                    for (const string &cmd : limitedHistory) {
                        cout << cmd << endl;
                    }
                }
            } else {
                pid_t child_pid = fork();
                if (child_pid == 0) {
                    int exec_status = execute_with_redirection(input_file, output_file, arguments, append_redirection);
                    if (exec_status != 0) {
                        exit(EXIT_FAILURE);
                    }
                } else if (child_pid > 0) {
                    int status;
                    waitpid(child_pid, &status, 0);
                } else {
                    perror("fork");
                }
            }

            if (strcmp(arguments[no_of_tokens - 1], "&") == 0) {
                run_command_background(input, bg_processes);
            } else {
                // run_command_foreground(input);
            }

            check_background_processes(bg_processes);
        }
    }

    return 0;
}
