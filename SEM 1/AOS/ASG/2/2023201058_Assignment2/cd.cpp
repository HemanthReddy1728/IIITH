#include "headers.h"
using namespace std;

// Function to change the directory (cd command)
void cd_command(const char *path) {
    if (path != nullptr) {
        if (strcmp(path, "-") == 0) {
            // Handle "cd -" to change to the previous working directory
            char current_directory[1024];
            if (getcwd(current_directory, sizeof(current_directory)) == nullptr) {
                perror("getcwd");
                return;
            }

            const char *previous_dir = getenv("OLDPWD");
            if (previous_dir != nullptr) {
                if (chdir(previous_dir) == 0) {
                    // Print the new current directory
                    if (getcwd(current_directory, sizeof(current_directory)) != nullptr) {
                        cout << current_directory << endl;
                    } else {
                        perror("getcwd");
                    }

                    // Update the environment variable OLDPWD
                    setenv("OLDPWD", current_directory, 1);
                } else {
                    perror("chdir");
                }
            } else {
                cerr << "OLDPWD not set" << endl;
            }
        } else if (strcmp(path, ".") == 0) {
            // Do nothing (stay in the current directory)
            return;
        } else if (strcmp(path, "..") == 0) {
            if (chdir("..") != 0) {
                perror("chdir");
            }
        } else if (strcmp(path, "~") == 0) {
            // Change to the user's home directory
            struct passwd *pw = getpwuid(getuid());
            if (pw != nullptr && pw->pw_dir != nullptr) {
                if (chdir(pw->pw_dir) != 0) {
                    perror("chdir");
                }
            } else {
                cerr << "Unable to determine home directory" << endl;
            }
        } else if (strncmp(path, "~/", 2) == 0) {
            // Expand "~/..." to the home directory path
            struct passwd *pw = getpwuid(getuid());
            if (pw != nullptr && pw->pw_dir != nullptr) {
                string home_path = pw->pw_dir;
                home_path += path + 1; // Remove the "~" and add the rest of the path
                if (chdir(home_path.c_str()) != 0) {
                    perror("chdir");
                }
            } else {
                cerr << "Unable to determine home directory" << endl;
            }
        } else if (path[0] == '/') {
            // Change to an absolute path
            if (chdir(path) != 0) {
                perror("chdir");
            }
        } else {
            // Change to a relative path
            char current_directory[1024];
            if (getcwd(current_directory, sizeof(current_directory)) != nullptr) {
                string new_directory = current_directory;
                new_directory += "/";
                new_directory += path;
                if (chdir(new_directory.c_str()) != 0) {
                    perror("chdir");
                }
            } else {
                perror("getcwd");
            }
        }

        // Update the environment variable OLDPWD
        char current_directory[1024];
        if (getcwd(current_directory, sizeof(current_directory)) != nullptr) {
            setenv("OLDPWD", current_directory, 1);
        } else {
            perror("getcwd");
        }
    } else {
        // No argument provided, change to the home directory
        // const char *home_path = getenv("HOME");
        // if (home_path != nullptr) {
        //     chdir(home_path);
        // } else {
        //     cerr << "HOME environment variable not set" << endl;
        // }

        struct passwd *pw = getpwuid(getuid());
        if (pw != nullptr && pw->pw_dir != nullptr) {
            if (chdir(pw->pw_dir) != 0) {
                perror("chdir");
            }

            // Update the environment variable OLDPWD
            char current_directory[1024];
            if (getcwd(current_directory, sizeof(current_directory)) != nullptr) {
                setenv("OLDPWD", current_directory, 1);
            } else {
                perror("getcwd");
            }
        } else {
            cerr << "Unable to determine home directory" << endl;
        }
    }
}


