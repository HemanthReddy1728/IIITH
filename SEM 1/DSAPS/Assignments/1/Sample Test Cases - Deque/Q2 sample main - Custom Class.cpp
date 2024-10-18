#include <iostream>

using namespace std;

// Your Deque Implementation
// class MyDeque{
//
// };

// Custom Class
class DiffPrinter {
   private:
    int x = 0;
    int y = 0;

   public:
    friend ostream &operator<<(ostream &output, const DiffPrinter &D)
    {
        output << abs(D.x - D.y);
        return output;
    }
    friend istream &operator>>(istream &input, DiffPrinter &D)
    {
        input >> D.x >> D.y;
        return input;
    }
};

int main()
{
    MyDeque<DiffPrinter> dq;
    DiffPrinter D;

    int choice = -1;
    while (choice != 0) {
        cin >> choice;
        if (choice == 1) {
            dq = MyDeque<DiffPrinter>();
        }
        else if (choice == 2) {
            int n;
            cin >> n;
            dq = MyDeque<DiffPrinter>(n);
        }
        else if (choice == 3) {
            int n;
            cin >> n;
            cin >> D;
            dq = MyDeque<DiffPrinter>(n, D);
        }
        else if (choice == 4) {
            cin >> D;
            cout << boolalpha << dq.push_back(D) << endl;
        }
        else if (choice == 5) {
            cout << boolalpha << dq.pop_back() << endl;
        }
        else if (choice == 6) {
            cin >> D;
            cout << boolalpha << dq.push_front(D) << endl;
        }
        else if (choice == 7) {
            cout << boolalpha << dq.pop_front() << endl;
        }
        else if (choice == 8) {
            D = dq.front();
            cout << D << endl;
        }
        else if (choice == 9) {
            D = dq.back();
            cout << D << endl;
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
