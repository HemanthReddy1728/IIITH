#include <bits/stdc++.h>

using namespace std;

int main() {
    long long a;
    ifstream fs("output.txt");
    long long prev;
    long long count = 0;

    fs >> a;
    prev = a;

    while (fs >> a) {
        count++;
        if (prev > a) {
            cout << "Not sorted" << endl;
            break;
        } else {
            prev = a;
        }
    }

    cout << "sorted hai bhai" << endl;
    cout << count << endl;
    fs.close();

    return 0;
}