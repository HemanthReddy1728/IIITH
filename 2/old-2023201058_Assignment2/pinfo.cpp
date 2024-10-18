#include "headers.h"
using namespace std;

void processInfo(const string& pid) {
    // Get the process ID (PID) of the current process
    pid_t current_pid = getpid();

    // Create a file path for the `/proc/<pid>/stat` file
    string stat_file_path;
    if (pid.empty()) {
        stringstream ss;
        ss << "/proc/" << current_pid << "/stat";
        stat_file_path = ss.str();
    } else {
        stat_file_path = "/proc/" + pid + "/stat";
    }

    // Read the `/proc/<pid>/stat` file
    ifstream stat_file(stat_file_path.c_str());
    if (!stat_file) {
        cerr << "Error: Unable to open stat file for PID " << pid << endl;
        return;
    }

    // Parse and extract process information
    string line;
    getline(stat_file, line);
    stat_file.close();

    stringstream linestream(line);
    string token;

    // Extract fields from the stat file
    int field_num = 0;
    string process_pid;
    string process_status;
    string process_memory;
    string process_executable;

    while (getline(linestream, token, ' ')) {
        ++field_num;
        if (field_num == 1) {
            process_pid = token;
        } else if (field_num == 3) {
            process_status = token;
        } else if (field_num == 23) {
            process_memory = token;
        } else if (field_num == 2) {
            process_executable = token;
        }
    }

    // Determine process status
    string status_code = "{" + process_status + "}";
    if (process_status == "R" || process_status == "S") {
        status_code += "+"; // Add '+' for running or sleeping processes
    }

    // Print process information
    cout << "pid -- " << process_pid << endl;
    cout << "Process Status -- " << status_code << endl;
    cout << "memory -- " << process_memory << " {Virtual Memory}" << endl;
    cout << "Executable Path -- " << process_executable << endl;
}