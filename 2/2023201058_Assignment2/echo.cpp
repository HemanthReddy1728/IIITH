#include "headers.h"
using namespace std;

// Function to implement the 'echo' command
void echo_command(char *input[]) {
    if (input == nullptr) {
        // cerr << "Invalid arguments" << endl;
        cerr << " " << endl;
        return;
    }

    vector<string> output_tokens; // Store the tokens to be printed

    // Iterate through the input tokens
    for (int i = 1; input[i] != nullptr; i++) {
        string token(input[i]);

        // Check if the token is enclosed in double quotes
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
            // Remove the double quotes and add the token to the output
            output_tokens.push_back(token.substr(1, token.size() - 2));
        } else {
            // Token is not enclosed in double quotes, add it as is
            output_tokens.push_back(token);
        }
    }

    // Print the tokens with spaces in between
    for (const string &token : output_tokens) {
        cout << token << ' ';
    }

    cout << endl; // Print a newline at the end
}