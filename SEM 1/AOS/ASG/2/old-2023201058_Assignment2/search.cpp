#include "headers.h"
using namespace std;

// Function to check if a file or directory exists
bool fileExistence(const string& path) {
    ifstream file(path);
    return file.good();
}

// Function to search for a file or folder recursively
bool searchFileOrFolderRecursively(const string& name, const string& path) {
    DIR* dir = opendir(path.c_str());

    if (dir == nullptr) {
        cerr << "Error opening directory: " << path << endl;
        return false;
    }

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        string entryName = entry->d_name;
        if (entryName == "." || entryName == "..") {
            continue;
        }

        string entryPath = path + "/" + entryName;

        struct stat entryStat;
        if (lstat(entryPath.c_str(), &entryStat) == -1) {
            cerr << "Error getting file information: " << entryPath << endl;
            continue;
        }

        if (S_ISDIR(entryStat.st_mode)) {
            if (searchFileOrFolderRecursively(name, entryPath)) {
                closedir(dir);
                return true;
            }
        } else if (S_ISREG(entryStat.st_mode)) {
            if (entryName == name) {
                closedir(dir);
                return true;
            }
        }
        
        if (entryName == name) {
            closedir(dir);
            return true;
        }

        if (fileExistence(entryPath)) {
            continue;
        }

        if (entry->d_type == DT_DIR) {
            if (searchFileOrFolderRecursively(name, entryPath)) {
                closedir(dir);
                return true;
            }
        }
    }

    closedir(dir);
    return false;
}

// Function to initiate the search
bool searchFileOrFolder(const string& name, const string& path) {
    return searchFileOrFolderRecursively(name, path);
}
// bool searchFileOrFolder(const string& name) {
//     return searchFileOrFolderRecursively(name, ".");
// }