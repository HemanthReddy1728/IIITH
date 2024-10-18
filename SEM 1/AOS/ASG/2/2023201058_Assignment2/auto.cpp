#include "headers.h"
using namespace std;

vector<string> listFilesInDirectory(const string& directory) {
    vector<string> files;
    DIR* dir = opendir(directory.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir))) {
            files.push_back(entry->d_name);
        }
        closedir(dir);
    }
    return files;
}

vector<string> autocomplete(const string& partial) {
    vector<string> suggestions;
    string currentDirectory = "."; // You can modify this to the desired directory

    // List all files and directories in the current directory
    vector<string> files = listFilesInDirectory(currentDirectory);

    // Check if the partial input matches any file or directory
    for (const string& file : files) {
        if (file.compare(0, partial.size(), partial) == 0) {
            suggestions.push_back(file);
        }
    }

    return suggestions;
}
