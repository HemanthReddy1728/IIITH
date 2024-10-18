#include "headers.h"
using namespace std;

// Function to check if a file or directory exists
bool fileExists(const string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Function to get file type and permissions in string format (e.g., "d-rwxr-xr-x")
string getFilePermissions(const struct stat& fileStat) {
    string permissions;

    // Determine file type
    if (S_ISREG(fileStat.st_mode)) {
        permissions += "-";  // Regular file
    } else if (S_ISDIR(fileStat.st_mode)) {
        permissions += "d";  // Directory
    } else if (S_ISLNK(fileStat.st_mode)) {
        permissions += "l";  // Symbolic link
    } else if (S_ISFIFO(fileStat.st_mode)) {
        permissions += "p";  // FIFO (named pipe)
    } else if (S_ISSOCK(fileStat.st_mode)) {
        permissions += "s";  // Socket
    } else if (S_ISCHR(fileStat.st_mode)) {
        permissions += "c";  // Character device
    } else if (S_ISBLK(fileStat.st_mode)) {
        permissions += "b";  // Block device
    } else {
        permissions += "?";  // Unknown file type
    }

    // Owner permissions
    permissions += (fileStat.st_mode & S_IRUSR) ? 'r' : '-';
    permissions += (fileStat.st_mode & S_IWUSR) ? 'w' : '-';
    permissions += (fileStat.st_mode & S_IXUSR) ? 'x' : '-';

    // Group permissions
    permissions += (fileStat.st_mode & S_IRGRP) ? 'r' : '-';
    permissions += (fileStat.st_mode & S_IWGRP) ? 'w' : '-';
    permissions += (fileStat.st_mode & S_IXGRP) ? 'x' : '-';

    // Others permissions
    permissions += (fileStat.st_mode & S_IROTH) ? 'r' : '-';
    permissions += (fileStat.st_mode & S_IWOTH) ? 'w' : '-';
    permissions += (fileStat.st_mode & S_IXOTH) ? 'x' : '-';

    return permissions;
}

// Function to list files and directories in the specified directory
void listFiles(const string& dirPath, bool showHidden) {
    DIR* dir = opendir(dirPath.c_str());

    if (dir == nullptr) {
        cerr << "Error opening directory: " << dirPath << endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        string fileName = entry->d_name;

        // Skip hidden files and directories if not requested
        if (!showHidden && fileName[0] == '.') {
            continue;
        }

        cout << fileName << " ";
    }

    cout << endl;
    closedir(dir);
}

// Function to list files and directories in the specified directory in long format (-l)
void listFilesLongFormat(const string& dirPath, bool showHidden) {
    DIR* dir = opendir(dirPath.c_str());

    if (dir == nullptr) {
        cerr << "Error opening directory: " << dirPath << endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        string fileName = entry->d_name;

        // Skip hidden files and directories if not requested
        if (!showHidden && fileName[0] == '.') {
            continue;
        }

        string filePath = dirPath + "/" + fileName;
        struct stat fileStat;

        if (stat(filePath.c_str(), &fileStat) < 0) {
            cerr << "Error getting file stats for: " << fileName << endl;
            continue;
        }

        struct passwd* pw = getpwuid(fileStat.st_uid);
        struct group* gr = getgrgid(fileStat.st_gid);
        struct tm* timeInfo = localtime(&fileStat.st_mtime);
        char timeStr[80];
        strftime(timeStr, sizeof(timeStr), "%b %d %H:%M", timeInfo);

        cout << getFilePermissions(fileStat) << " ";
        cout << fileStat.st_nlink << " ";
        cout << (pw ? pw->pw_name : "unknown") << " ";
        cout << (gr ? gr->gr_name : "unknown") << " ";
        cout << setw(5) << fileStat.st_size << " ";
        cout << timeStr << " ";
        cout << fileName << endl;
    }

    cout << endl;
    closedir(dir);
}

// Function to list a single file in long format (-l)
void listFileLongFormat(const string& filePath) {
    struct stat fileStat;
    if (stat(filePath.c_str(), &fileStat) < 0) {
        cerr << "Error getting file stats for: " << filePath << endl;
        return;
    }

    struct passwd* pw = getpwuid(fileStat.st_uid);
    struct group* gr = getgrgid(fileStat.st_gid);
    struct tm* timeInfo = localtime(&fileStat.st_mtime);
    char timeStr[80];
    strftime(timeStr, sizeof(timeStr), "%b %d %H:%M", timeInfo);

    cout << getFilePermissions(fileStat) << " ";
    cout << fileStat.st_nlink << " ";
    cout << (pw ? pw->pw_name : "unknown") << " ";
    cout << (gr ? gr->gr_name : "unknown") << " ";
    cout << setw(5) << fileStat.st_size << " ";
    cout << timeStr << " ";
    cout << filePath << endl;
}

void listFile(const string& filePath) {
    cout << filePath << endl;
}


void listDirectory(const vector<string>& args) {
    bool showHidden = false;
    bool longFormat = false;
    vector<string> directories;

    // Parse flags and directories
    for (size_t i = 1; i < args.size(); ++i) {
        const string& arg = args[i];
        if (arg == "-a") {
            showHidden = true;
        } else if (arg == "-l") {
            longFormat = true;
        } else if (arg == "-la" || arg == "-al") {
            longFormat = true;
            showHidden = true;
        } else if (arg[0] == '-') {
                cerr << "Invalid flag: " << arg << endl;
            return;
        } else {
            directories.push_back(arg);
        }
    }

    if (directories.empty()) {
        // No directories specified, list the current directory
        directories.push_back(".");
    }

    // for (const string& dir : directories) {
    //     if (dir == "~") {
    //         const char* homeDir = getenv("HOME");
    //         if (homeDir) {
    //             listFilesLongFormat(homeDir, showHidden);
    //         } else {
    //             cerr << "Home directory not found." << endl;
    //         }
    //     } else if (!fileExists(dir)) {
    //         cerr << "Directory or file not found: " << dir << endl;
    //         continue;

    //     } else if (longFormat) {
    //         // cout << "Listing directory: " << dir << " (long format)" << endl;
    //         listFilesLongFormat(dir, showHidden);
    //     } else {
    //         // cout << "Listing directory: " << dir << endl;
    //         listFiles(dir, showHidden);
    //     }
    // }

    for (const string& path : directories) {
        if (path == "~") {
            const char* homeDir = getenv("HOME");
            if (homeDir) {
                listFilesLongFormat(homeDir, showHidden);
            } else {
                cerr << "Home directory not found." << endl;
            }
            continue;
        }

        struct stat fileStat;
        if (stat(path.c_str(), &fileStat) < 0) {
            cerr << "Error getting file stats for: " << path << endl;
            continue;
        }

        if (S_ISDIR(fileStat.st_mode)) {
            // It's a directory
            if (longFormat) {
                listFilesLongFormat(path, showHidden);
            } else {
                listFiles(path, showHidden);
            }
        } else if (S_ISREG(fileStat.st_mode)) {
            // It's a regular file
            if (longFormat) {
                listFileLongFormat(path);
            } else {
                listFile(path);
            }
        } else {
            cerr << "Unknown file type: " << path << endl;
        }
    }
}



