#include "headers.h"
using namespace std;

// Define a global variable to keep track of the foreground process PID
static pid_t foreground_pid = -1;

class SignalHandler {
public:
    SignalHandler();
    void RegisterHandlers();
    static void CtrlZHandler(int signo);
    static void CtrlCHandler(int signo);
    static void ChildHandler(int signo);
    static void SendSignal(int pid, int signo);
};

SignalHandler::SignalHandler() {
    // Constructor, if needed
}

void SignalHandler::RegisterHandlers() {
    // Register signal handlers
    signal(SIGTSTP, CtrlZHandler); // Handle Ctrl-Z
    signal(SIGINT, CtrlCHandler);  // Handle Ctrl-C
    signal(SIGQUIT, SIG_IGN);     // Ignore Ctrl-\ (SIGQUIT)
    signal(SIGTERM, SIG_IGN);     // Ignore termination signal
    signal(SIGTTIN, SIG_IGN);     // Ignore background process terminal input
    signal(SIGTTOU, SIG_IGN);     // Ignore background process terminal output
    signal(SIGCHLD, ChildHandler); // Handle child process termination
}

void SignalHandler::CtrlZHandler(int signo) {
    if (foreground_pid > 0) {
        // Send SIGTSTP to the foreground process to stop it
        SendSignal(foreground_pid, SIGTSTP);
        foreground_pid = -1; // No foreground process running
    }
}

void SignalHandler::CtrlCHandler(int signo) {
    if (foreground_pid > 0) {
        // Send SIGINT to the foreground process to interrupt it
        SendSignal(foreground_pid, SIGINT);
        foreground_pid = -1; // No foreground process running
    }
}

void SignalHandler::ChildHandler(int signo) {
    // Handle child process termination
    int status;
    pid_t pid = waitpid(-1, &status, WNOHANG);
    if (pid > 0) {
        // Child process with PID 'pid' has terminated
        if (foreground_pid == pid) {
            foreground_pid = -1; // Reset foreground PID
        }
        // You can add additional handling logic here if needed.
    }
}

void SignalHandler::SendSignal(int pid, int signo) {
    // Send a signal to a specific process
    kill(pid, signo);
}

