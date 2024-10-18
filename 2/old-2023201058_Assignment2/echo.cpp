#include "headers.h"
using namespace std;

void echoCommand(const std::vector<std::string>& args) {
    for (size_t i = 1; i < args.size(); ++i) {
        if (i > 1) {
            std::cout << ' ';
        }
        std::cout << args[i];
    }
    std::cout << std::endl;
}
