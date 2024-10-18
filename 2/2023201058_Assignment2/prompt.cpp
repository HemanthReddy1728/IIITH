#include "headers.h"
using namespace std;

// Function to get the absolute path of the home directory
char* get_absolute_home_path() {
    struct passwd *pw = getpwuid(getuid());
    if (pw != NULL && pw->pw_dir != NULL) {
        return strdup(pw->pw_dir);
    } else {
        fprintf(stderr, "Unable to determine home directory\n");
        return NULL;
    }
}

// Function to display the shell prompt
void display_prompt() {
    // Get the username and system name
    char *username = getenv("USER");
    struct utsname system_info;
    if (uname(&system_info) == -1) {
        perror("uname");
        return;
    }

    // Get the current working directory
    char current_directory[1024];
    if (getcwd(current_directory, sizeof(current_directory)) == NULL) {
        perror("getcwd");
        return;
    }

    // Get the absolute path of the home directory
    char *home_directory = get_absolute_home_path();

    // Check if the current directory is the same as the home directory
    if (strcmp(current_directory, home_directory) == 0) {
        // Set current directory as the absolute path of the home directory
        strcpy(current_directory, home_directory);
    }

    // Display the shell prompt with the absolute path of the current directory
    printf("%s@%s:%s> ", username, system_info.nodename, current_directory);
    fflush(stdout);  // Flush the output buffer to ensure prompt is displayed immediately

    // Free the allocated memory
    free(home_directory);
}
