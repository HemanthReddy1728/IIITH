#include "headers.h"
using namespace std;

std::vector<std::string> listFilesInDirectory(const std::string& dirPath) {
    std::vector<std::string> files;
    DIR* dir = opendir(dirPath.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir))) {
            if (entry->d_type == DT_DIR || entry->d_type == DT_REG) {
                files.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }
    return files;
}

void autoComplete(std::string& input) {
    std::string partial;
    size_t lastSpace = input.find_last_of(" ");
    if (lastSpace != std::string::npos) {
        partial = input.substr(lastSpace + 1);
    } else {
        partial = input;
    }

    std::vector<std::string> files = listFilesInDirectory(".");
    std::vector<std::string> matches;
    for (const std::string& file : files) {
        if (file.find(partial) == 0) {
            matches.push_back(file);
        }
    }

    if (matches.size() == 1) {
        input = input.substr(0, lastSpace + 1) + matches[0];
    } else if (matches.size() > 1) {
        std::string suggestions;
        for (const std::string& match : matches) {
            suggestions += match + " ";
        }
        std::cout << std::endl << suggestions << std::endl;
    }
}
