#include <iostream>
#include <string>

using namespace std;

/*
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
};*/

template <typename T>
class MyDeque {
private:
    T* Arr;
    long long Cpct;
    long long Frontindex;
    long long Rearindex;
    long long MyDequesize;

public:

    void printdetails()
    {
        return; // Comment this return statement to use printdetails function
        cout << Cpct << " " << Frontindex << " " << Rearindex << " " << MyDequesize << endl;
        if (Frontindex == -1 && Rearindex == -1)
        {
            return;
        }
        long long f = Frontindex;
        if (MyDequesize)
        {
            for (long long i = 0; i < MyDequesize; i++, f = (f + 1) % Cpct)
            {
                cout << Arr[f] << " ";
            }
        }
        cout << endl;
        return;
    } 

    MyDeque() : Cpct(0), Frontindex(-1), Rearindex(-1), MyDequesize(0) 
    {
        Arr = new T[Cpct];
        printdetails();
    }

    MyDeque(long long n) : Cpct(n), Frontindex(0), Rearindex(n - 1), MyDequesize(n) 
    {
        Arr = new T[Cpct];
        for (long long i = 0; i < n; i++) 
        {
            Arr[i] = T();
        }
        printdetails();
    }

    MyDeque(long long n, T x) : Cpct(n), Frontindex(0), Rearindex(n - 1), MyDequesize(n) 
    {
        Arr = new T[Cpct];
        for (long long i = 0; i < n; i++) 
        {
            Arr[i] = x;
        }
        printdetails();
    }

    // Destructor creates seg faults
    // ~MyDeque() 
    // {
    //     delete[] Arr;
    // }

    void reserve(long long n) 
    {
        if (n > Cpct) 
        {
            T* newArr = new T[n];
            for (long long i = 0; i < MyDequesize; i++) 
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            Cpct = n;
            Frontindex = 0;
            Rearindex = MyDequesize - 1;
            // delete[] Arr;
            Arr = newArr;
        }
        printdetails();
    }

    bool push_back(T x) 
    {
        if (Frontindex == Rearindex && Rearindex == -1)
        {
            T* arr1 = new T[1];
            arr1[0] = x;
            // delete [] Arr;
            Arr = arr1;
            Frontindex = 0;
            Rearindex = 0;
            MyDequesize++;
            Cpct = 1;
            printdetails();
            return true;
        }

        if (full()) 
        {
            if (MyDequesize == Cpct) 
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
        }
        if (empty()) 
        {
            Frontindex = Rearindex = 0;
        } 
        else 
        {
            Rearindex = (Rearindex + 1) % Cpct;
        }
        Arr[Rearindex] = x;
        MyDequesize++;
        printdetails();
        return true;
    }

    bool pop_back() 
    {
        if (empty()) 
        {
            return false;
        }
        if (Frontindex == Rearindex) 
        {
            Frontindex = Rearindex = -1;
        } 
        else 
        {
            Rearindex = (Rearindex - 1 + Cpct) % Cpct;
        }
        MyDequesize--;
        printdetails();
        return true;
    }

    bool push_front(T x) 
    {
        if (Frontindex == Rearindex && Rearindex == -1)
        {
            T* arr1 = new T[1];
            arr1[0] = x;
            // delete [] Arr;
            Arr = arr1;
            Frontindex = 0;
            Rearindex = 0;
            MyDequesize++;
            Cpct = 1;
            printdetails();
            return true;
        }
        if (full()) 
        {
            if (MyDequesize == Cpct) 
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
        }
        if (empty()) 
        {
            Frontindex = Rearindex = 0;
        } 
        else 
        {
            Frontindex = (Frontindex - 1 + Cpct) % Cpct;
        }
        Arr[Frontindex] = x;
        MyDequesize++;
        printdetails();
        return true;
    }

    bool pop_front() 
    {
        if (empty()) 
        {
            return false;
        }
        if (Frontindex == Rearindex) 
        {
            Frontindex = Rearindex = -1;
        } 
        else 
        {
            Frontindex = (Frontindex + 1) % Cpct;
        }
        MyDequesize--;
        printdetails();
        return true;
    }

    T front() 
    {
        if (empty()) 
        {
            return T(); 
        }
        return Arr[Frontindex];
    }

    T back() 
    {
        if (empty()) 
        {
            return T(); 
        }
        return Arr[Rearindex];
    }

    T operator[](long long n) 
    {
        if (n >= 0) 
        {
            if (n >= MyDequesize) 
            {
                printdetails();
                return T(); 
            }
            printdetails();
            return Arr[(Frontindex + n) % Cpct];
        } 
        else 
        {
            n = -n;
            if (n > MyDequesize) 
            {
                printdetails();
                return T(); 
            }
            printdetails();
            return Arr[(Rearindex - n + 1 + Cpct) % Cpct];
        }
    }

    bool empty() 
    {
        printdetails();
        return MyDequesize == 0;
    }

    bool full() 
    {
        printdetails();
        return MyDequesize == Cpct;
    }

    long long size() 
    {
        printdetails();
        return MyDequesize;
    }

    void clear() 
    {
        Frontindex = Rearindex = -1;
        // MyDequesize = 0;
        printdetails();
    }


    long long capacity() 
    {
        printdetails();
        return Cpct;
    }

    void resize(long long n) 
    {
        if (n >= MyDequesize) 
        {
            long long oldSize = MyDequesize;
            MyDequesize = n;
            if (n > Cpct) 
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
            for (long long i = oldSize; i < MyDequesize; i++) 
            {
                Arr[(Frontindex + i) % Cpct] = T();
            }
        } 
        else 
        {
            MyDequesize = n;
            if (MyDequesize <= Cpct / 4) 
            {
                reserve(Cpct / 2);
            }
        }
        printdetails();
    }

    void resize(long long n, T d) 
    {
        if (n >= MyDequesize) 
        {
            long long oldSize = MyDequesize;
            MyDequesize = n;
            if (n > Cpct) 
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
            for (long long i = oldSize; i < MyDequesize; ++i) 
            {
                Arr[(Frontindex + i) % Cpct] = d;
            }
        } 
        else 
        {
            MyDequesize = n;
            if (MyDequesize <= Cpct / 4) 
            {
                reserve(Cpct / 2);
            }
        }
        printdetails();
    }


    void shrink_to_fit() 
    {
        if (Cpct > MyDequesize) 
        {
            T* newArr = new T[MyDequesize];
            for (long long i = 0; i < MyDequesize; i++) 
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            Cpct = MyDequesize;
            Frontindex = 0;
            Rearindex = MyDequesize - 1;
            // delete[] Arr;
            Arr = newArr;
        }
        printdetails();
    }
};



int main()
{
    
    MyDeque<string> dq;
    string s;

    int choice = -1;
    while (choice != 0) 
    {
        cin >> choice;
        if (choice == 1) 
        {
            dq = MyDeque<string>();
        }
        else if (choice == 2) 
        {
            int n;
            cin >> n;
            dq = MyDeque<string>(n);
        }
        else if (choice == 3) 
        {
            int n;
            cin >> n;
            cin >> s;
            dq = MyDeque<string>(n, s);
        }
        else if (choice == 4) 
        {
            cin >> s;
            cout << boolalpha << dq.push_back(s) << endl;
        }
        else if (choice == 5) 
        {
            cout << boolalpha << dq.pop_back() << endl;
        }
        else if (choice == 6) 
        {
            cin >> s;
            cout << boolalpha << dq.push_front(s) << endl;
        }
        else if (choice == 7) 
        {
            cout << boolalpha << dq.pop_front() << endl;
        }
        else if (choice == 8) 
        {
            s = dq.front();
            cout << s << endl;
        }
        else if (choice == 9) 
        {
            s = dq.back();
            cout << s << endl;
        }
        else if (choice == 10)
        {
            long long index;
            cin >> index;
            cout << dq[index] << endl;
        }
        else if (choice == 11) 
        {
            cout << boolalpha << dq.empty() << endl;
        }
        else if (choice == 12) 
        {
            cout << dq.size() << endl;
        }
        else if (choice == 13)
        {
            long long newSize;
            cin >> newSize;
            dq.resize(newSize);
        }
        else if (choice == 14)
        {
            long long newSize;
            string value;
            cin >> newSize;
            cin >> value;
            dq.resize(newSize, value);
        }
        else if (choice == 15)
        {
            long long newCapacity;
            cin >> newCapacity;
            dq.reserve(newCapacity);
        }
        else if (choice == 16)
        {
            dq.shrink_to_fit();
        }
        else if (choice == 17)
        {
            dq.clear();
        }
        else if (choice == 18) 
        {
            cout << dq.capacity() << endl;
        }
       
    }
    return 0;
}

/*
template <typename T> void Menu()
{
    Deque<T> d;
    // string dataType;
    // cout << "Give datatype of your deque : ";
    // cin >> dataType;
    // Deque<decltype(dataType)> d;

    while (true) 
    {
        cout << "Menu:" << endl;
        cout << "0. Exit" << endl;
        cout << "1. Empty Deque Initialization" << endl;
        cout << "2. Length 'n' Deque Initialization with dafault value of T datatype" << endl;
        cout << "3. Length 'n' Deque Initialization with all values as 'x'" << endl;
        cout << "4. Push Back" << endl;
        cout << "5. Pop Back" << endl;
        cout << "6. Push Front" << endl;
        cout << "7. Pop Front" << endl;
        cout << "8. Front" << endl;
        cout << "9. Back" << endl;
        cout << "10. T D[n] nth element" << endl;
        cout << "11. Empty" << endl;
        cout << "12. Size" << endl;
        cout << "13. Resize with Default Value" << endl;
        cout << "14. Resize with User-defined Value" << endl;
        cout << "15. Reserve" << endl;
        cout << "16. Shrink to Fit" << endl;
        cout << "17. Clear" << endl;
        cout << "18. Capacity" << endl;
        d.printdetails();
        cout << "Enter your choice: ";
        int choice;
        cin >> choice;
        system("clear");

        if (choice == 0) 
        {
            cout << "Exiting the program." << endl;
            return;
        }

        switch (choice) 
        {
            case 1:
                d = Deque<T>();
                // Deque<T> d;
                break;

            case 2:
                long long n;
                cout << "Length of the deque : ";
                cin >> n;
                d = Deque<T>(n);
                // Deque<T>(n) d;
                break;

            case 3:
                // long long n;
                T x;
                cout << "Length of the deque : ";
                cin >> n;
                cout << "Default value of the deque : ";
                cin >> x;
                d = Deque<T>(n, x);
                // Deque<T>(n, x) d;
                break;

            case 4: 
                T value;
                cout << "Enter value to push back: ";
                cin >> value;
                cout << d.push_back(value);
                break;
            
            case 5:
                cout << d.pop_back();
                break;

            case 6: 
                // long long value;
                cout << "Enter value to push front: ";
                cin >> value;
                cout << d.push_front(value);
                break;
            
            case 7:
                cout << d.pop_front();
                break;

            case 8:
                cout << "Front: " << d.front() << endl;
                break;

            case 9:
                cout << "Back: " << d.back() << endl;
                break;

            case 10: 
                long long index;
                cout << "Enter index: ";
                cin >> index;
                cout << "Value at index " << index << ": " << d[index] << endl;
                break;
            
            case 11:
                cout << "Is empty: " << (d.empty() ? "True" : "False") << endl;
                break;

            case 12:
                cout << "Size: " << d.size() << endl;
                break;

            case 13: 
                long long newSize;
                cout << "Enter new size: ";
                cin >> newSize;
                d.resize(newSize);
                break;
            
            case 14: 
                // long long newSize;
                // long long value;
                cout << "Enter new size: ";
                cin >> newSize;
                cout << "Enter value: ";
                cin >> value;
                d.resize(newSize, value);
                break;
            
            case 15: 
                long long newCapacity;
                cout << "Enter new capacity: ";
                cin >> newCapacity;
                d.reserve(newCapacity);
                break;
            
            case 16:
                d.shrink_to_fit();
                break;

            case 17:
                d.clear();
                // cout << "Cleared." << endl;
                break;

            case 18:
                cout << "Capacity: " << d.capacity() << endl;
                break;

            default:
                cout << "Invalid choice. Please enter a valid option." << endl;
                break;
        }
    }
}

int main() 
{
    // Deque<int> dq;
    // cout<<"hi";
    // dq.push_back(1);
    // dq.push_front(2);
 
    string dataType;
    cout << "Give datatype of your deque : ";
    cin >> dataType;

    // Menu<decltype(dataType)>();

    if (dataType == "int")
    {
        Menu<int>();
    }
    else if (dataType == "char")
    {
        Menu<char>();
    }
    else if (dataType == "bool")
    {
        Menu<bool>();
    }
    else if (dataType == "float")
    {
        Menu<float>();
    }
    else if (dataType == "double")
    {
        Menu<double>();
    }
    // else if (datatype == "void")
    // {
    //     Menu<void>();
    // }
    // else if (datatype == "wchar_t")
    // {
    //     Menu<wchar_t>();
    // }
    else 
    {
        cout << "Unfit datatype" << endl;
    }

    

    return 0;
}
*/
/*
int main()
{
    Deque<DiffPrinter> dq;
    DiffPrinter D;

    int choice = -1;
    while (choice != 0) 
    {
        cin >> choice;
        if (choice == 1) 
        {
            dq = Deque<DiffPrinter>();
        }
        else if (choice == 2) 
        {
            int n;
            cin >> n;
            dq = Deque<DiffPrinter>(n);
        }
        else if (choice == 3) 
        {
            int n;
            cin >> n;
            cin >> D;
            dq = Deque<DiffPrinter>(n, D);
        }
        else if (choice == 4) 
        {
            cin >> D;
            cout << boolalpha << dq.push_back(D) << endl;
        }
        else if (choice == 5) 
        {
            cout << boolalpha << dq.pop_back() << endl;
        }
        else if (choice == 6) 
        {
            cin >> D;
            cout << boolalpha << dq.push_front(D) << endl;
        }
        else if (choice == 7) 
        {
            cout << boolalpha << dq.pop_front() << endl;
        }
        else if (choice == 8) 
        {
            D = dq.front();
            cout << D << endl;
        }
        else if (choice == 9) 
        {
            D = dq.back();
            cout << D << endl;
        }
        else if (choice == 10)
        {
            long long index;
            cin >> index;
            cout << dq[index] << endl;
        }
        else if (choice == 11) 
        {
            cout << boolalpha << dq.empty() << endl;
        }
        else if (choice == 12) 
        {
            cout << dq.size() << endl;
        }
        else if (choice == 13)
        {
            long long newSize;
            cin >> newSize;
            dq.resize(newSize);
        }
        else if (choice == 14)
        {
            long long newSize;
            DiffPrinter value;
            cin >> newSize;
            cin >> value;
            dq.resize(newSize, value);
        }
        else if (choice == 15)
        {
            long long newCapacity;
            cin >> newCapacity;
            dq.reserve(newCapacity);
        }
        else if (choice == 16)
        {
            dq.shrink_to_fit();
        }
        else if (choice == 17)
        {
            dq.clear();
        }
        else if (choice == 18) 
        {
            cout << dq.capacity() << endl;
        }
        else 
        {
            // implement rest of the logic
        }
    }
    return 0;
}
*/