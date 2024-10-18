#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

template <typename KeyType, typename ValueType>
struct KeyValuePair
{
    KeyType key;
    ValueType value;
    KeyValuePair *next;

    KeyValuePair(KeyType k, ValueType v) : key(k), value(v), next(nullptr) {}
};


template <typename KeyType, typename ValueType>
class UnorderedMap
{
private:
    int TABLE_SIZE;  
    int numElements; 
    KeyValuePair<KeyType, ValueType> **table;

    int myHash(KeyType key)
    {
        ostringstream os;
        os << key;
        string keyString = os.str();

        int hashValue = 0;
        for (char c : keyString)
        {
            hashValue = (static_cast<int>(c) + hashValue * 31) % TABLE_SIZE;
        }

        hashValue = (hashValue >= 0) ? hashValue : -hashValue;

        return hashValue;
    }

    void resizeTable()
    {
        int newTableSize = TABLE_SIZE * 2;
        KeyValuePair<KeyType, ValueType> **newTable = new KeyValuePair<KeyType, ValueType> *[newTableSize];

        for (int i = 0; i < newTableSize; i++)
        {
            newTable[i] = nullptr;
        }

        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValuePair<KeyType, ValueType> *current = table[i];
            while (current)
            {
                int newIndex = myHash(current->key);
                KeyValuePair<KeyType, ValueType> *next = current->next;
                current->next = newTable[newIndex];
                newTable[newIndex] = current;
                current = next;
            }
        }

        delete[] table;
        TABLE_SIZE = newTableSize;
        table = newTable;
    }

public:
    UnorderedMap()
    {
        TABLE_SIZE = 10007; 
        numElements = 0;
        table = new KeyValuePair<KeyType, ValueType> *[TABLE_SIZE];

        for (int i = 0; i < TABLE_SIZE; i++)
        {
            table[i] = nullptr;
        }
    }

    bool insert(KeyType key, ValueType value)
    {
        if (numElements >= TABLE_SIZE)
        {
            resizeTable();
        }

        int index = myHash(key);
        KeyValuePair<KeyType, ValueType> *newNode = new KeyValuePair<KeyType, ValueType>(key, value);

        if (!table[index])
        {
            table[index] = newNode;
        }
        else
        {
            KeyValuePair<KeyType, ValueType> *current = table[index];
            while (current->next)
            {
                if (current->key == key)
                {
                    delete newNode;
                    return false; 
                }
                current = current->next;
            }
            if (current->key == key)
            {
                delete newNode;
                return false; 
            }
            current->next = newNode;
        }

        numElements++;
        return true;
    }

    bool erase(KeyType key)
    {
        int index = myHash(key);

        if (!table[index])
        {
            return false; 
        }

        KeyValuePair<KeyType, ValueType> *current = table[index];
        KeyValuePair<KeyType, ValueType> *prev = nullptr;

        while (current)
        {
            if (current->key == key)
            {
                if (prev)
                {
                    prev->next = current->next;
                }
                else
                {
                    table[index] = current->next;
                }
                delete current;
                numElements--;
                return true;
            }
            prev = current;
            current = current->next;
        }

        return false; 
    }

    bool contains(KeyType key)
    {
        int index = myHash(key);

        KeyValuePair<KeyType, ValueType> *current = table[index];
        while (current)
        {
            if (current->key == key)
            {
                return true;
            }
            current = current->next;
        }

        return false;
    }

    ValueType &operator[](KeyType key)
    {
        if (numElements >= TABLE_SIZE)
        {
            resizeTable();
        }

        int index = myHash(key);
        KeyValuePair<KeyType, ValueType> *current = table[index];

        while (current)
        {
            if (current->key == key)
            {
                return current->value;
            }
            current = current->next;
        }

        KeyValuePair<KeyType, ValueType> *newNode = new KeyValuePair<KeyType, ValueType>(key, ValueType());
        newNode->next = table[index];
        table[index] = newNode;
        numElements++;
        return table[index]->value;
    }

    void clear()
    {
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValuePair<KeyType, ValueType> *current = table[i];
            while (current)
            {
                KeyValuePair<KeyType, ValueType> *temp = current;
                current = current->next;
                delete temp;
            }
            table[i] = nullptr;
        }
        numElements = 0;
    }

    int size()
    {
        return numElements;
    }

    bool empty()
    {
        return numElements == 0;
    }

    vector<KeyType> keys()
    {
        vector<KeyType> result;
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValuePair<KeyType, ValueType> *current = table[i];
            while (current)
            {
                result.push_back(current->key);
                current = current->next;
            }
        }
        return result;
    }
};

int main()
{
    UnorderedMap<string, int> myMap;

    while (true)
    {
        int choice;
        cin >> choice;

        string key;
        int value;
        vector<string> keys = myMap.keys();

        if (choice == 0)
        {
            break;
        }

        switch (choice)
        {
        case 1:
        {
            cin >> key >> value;
            cout << (myMap.insert(key, value) ? "true" : "false") << endl;
            break;
        }
        case 2:
        {
            cin >> key;
            cout << (myMap.erase(key) ? "true" : "false") << endl;
            break;
        }
        case 3:
        {
            cin >> key;
            cout << (myMap.contains(key) ? "true" : "false") << endl;
            break;
        }
        case 4:
        {
            cin >> key;
            cout << myMap[key] << endl;
            break;
        }
        case 5:
            myMap.clear();
            break;
        case 6:
            cout << myMap.size() << endl;
            break;
        case 7:
            cout << (myMap.empty() ? "true" : "false") << endl;
            break;
        case 8:
        {
            for (int i = 0; i < keys.size(); i++)
            {
                cout << keys[i] << endl;
            }
            break;
        }
        default:
            cout << "Invalid choice. Please try again." << endl;
        }
    }

    return 0;
}
