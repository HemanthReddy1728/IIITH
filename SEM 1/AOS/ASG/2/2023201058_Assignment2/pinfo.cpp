#include "headers.h"
using namespace std;


void pinfo_command(const char* pid_str = nullptr) {
    pid_t pid;
    string status;
    string memory;
    string executable_path;

    if (pid_str == nullptr) {
        // If no PID is provided, use the PID of the current process (the shell)
        pid = getpid();
    } else {
        // If a PID is provided, convert it to an integer
        istringstream ss(pid_str);
        if (!(ss >> pid)) {
            cerr << "Invalid PID: " << pid_str << endl;
            return;
        }
    }

    // Read process status from /proc/<pid>/status
    ifstream status_file("/proc/" + to_string(pid) + "/status");
    if (status_file.is_open()) {
        string line;
        while (getline(status_file, line)) {
            if (line.find("State:") != string::npos) {
                status = line.substr(line.find(":") + 2, 1);
            } else if (line.find("VmSize:") != string::npos) {
                memory = line.substr(line.find(":") + 2);
            }
        }
        status_file.close();
    } else {
        cerr << "Unable to open status file for PID " << pid << endl;
        return;
    }

    // Read executable path from /proc/<pid>/exe
    char exe_path[4096];
    ssize_t exe_path_len = readlink(("/proc/" + to_string(pid) + "/exe").c_str(), exe_path, sizeof(exe_path));
    if (exe_path_len != -1) {
        exe_path[exe_path_len] = '\0';
        executable_path = exe_path;
    } else {
        cerr << "Unable to read executable path for PID " << pid << endl;
        return;
    }

    // Print process information
    cout << "pid -- " << pid << endl;
    cout << "Process Status -- {" << status << "}" << endl;
    cout << "memory -- " << memory << " {Virtual Memory}" << endl;
    cout << "Executable Path -- " << executable_path << endl;
}



/*
void pinfo_command(const char* pid_str = nullptr) {
    pid_t pid;
    string status;
    string memory;
    string executable_path;

    if (pid_str == nullptr) {
        // If no PID is provided, use the PID of the current process (the shell)
        pid = getpid();
    } else {
        // If a PID is provided, convert it to an integer
        istringstream ss(pid_str);
        if (!(ss >> pid)) {
            cerr << "Invalid PID: " << pid_str << endl;
            return;
        }
    }

    // Read process status from /proc/<pid>/status
    ifstream status_file("/proc/" + to_string(pid) + "/status");
    if (status_file.is_open()) {
        string line;
        while (getline(status_file, line)) {
            if (line.find("State:") != string::npos) {
                status = line.substr(line.find(":") + 2, 1);
            } else if (line.find("VmSize:") != string::npos) {
                memory = line.substr(line.find(":") + 2);
            }
        }
        status_file.close();
    } else {
        cerr << "Unable to open status file for PID " << pid << endl;
        return;
    }

    // Read executable path from /proc/<pid>/exe
    char exe_path[4096];
    ssize_t exe_path_len = readlink(("/proc/" + to_string(pid) + "/exe").c_str(), exe_path, sizeof(exe_path));
    if (exe_path_len != -1) {
        exe_path[exe_path_len] = '\0';
        executable_path = exe_path;
    } else {
        cerr << "Unable to read executable path for PID " << pid << endl;
        return;
    }

    // Print process information
    cout << "pid -- " << pid << endl;
    cout << "Process Status -- {" << status << "}" << endl;
    cout << "memory -- " << memory << " {Virtual Memory}" << endl;
    cout << "Executable Path -- " << executable_path << endl;
}

*/
