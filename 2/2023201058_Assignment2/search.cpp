#include "headers.h"
using namespace std;

// Function to recursively search for a file or folder
bool search_recursive(const string& path, const string& target) {
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        // Directory cannot be opened, return false
        return false;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        string full_path = path + "/" + entry->d_name;
        struct stat file_stat;
        if (stat(full_path.c_str(), &file_stat) == -1) {
            continue;
        }

        if (S_ISDIR(file_stat.st_mode)) {
            // It's a directory, recursively search
            if (search_recursive(full_path, target)) {
                closedir(dir);
                return true;
            }
        } else if (strcmp(entry->d_name, target.c_str()) == 0) {
            // Found the target file or folder
            closedir(dir);
            return true;
        }
    }

    closedir(dir);
    return false;
}


