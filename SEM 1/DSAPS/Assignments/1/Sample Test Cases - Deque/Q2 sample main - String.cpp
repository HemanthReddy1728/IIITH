#include <iostream>
#include <string>


using namespace std;

// Your Deque Implementation
// class MyDeque{
//
// };

int main()
{
    MyDeque<string> dq;
    string s;

    int choice = -1;
    while (choice != 0) {
        cin >> choice;
        if (choice == 1) {
            dq = MyDeque<string>();
        }
        else if (choice == 2) {
            int n;
            cin >> n;
            dq = MyDeque<string>(n);
        }
        else if (choice == 3) {
            int n;
            cin >> n;
            cin >> s;
            dq = MyDeque<string>(n, s);
        }
        else if (choice == 4) {
            cin >> s;
            cout << boolalpha << dq.push_back(s) << endl;
        }
        else if (choice == 5) {
            cout << boolalpha << dq.pop_back() << endl;
        }
        else if (choice == 6) {
            cin >> s;
            cout << boolalpha << dq.push_front(s) << endl;
        }
        else if (choice == 7) {
            cout << boolalpha << dq.pop_front() << endl;
        }
        else if (choice == 8) {
            s = dq.front();
            cout << s << endl;
        }
        else if (choice == 9) {
            s = dq.back();
            cout << s << endl;
        }
        else if (choice == 11) {
            cout << boolalpha << dq.empty() << endl;
        }
        else if (choice == 12) {
            cout << dq.size() << endl;
        }
        else if (choice == 18) {
            cout << dq.capacity() << endl;
        }
        else {
            // implement rest of the logic
        }
    }
    return 0;
}
