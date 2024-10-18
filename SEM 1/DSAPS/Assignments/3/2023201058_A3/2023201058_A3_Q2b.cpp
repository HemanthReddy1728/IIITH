#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

template <typename T1, typename T2>
struct KeyValue
{
    T1 key;
    T2 value;
    KeyValue *next;

    KeyValue(T1 k, T2 v) : key(k), value(v), next(nullptr) {}
};

template <typename T1, typename T2>
class UnorderedMap
{
private:
    int TABLE_SIZE; 
    int numElements;
    KeyValue<T1, T2> **table;

    int myHash(T1 key)
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
        KeyValue<T1, T2> **newTable = new KeyValue<T1, T2> *[newTableSize];

        for (int i = 0; i < newTableSize; i++)
        {
            newTable[i] = nullptr;
        }

        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValue<T1, T2> *current = table[i];
            while (current)
            {
                int newIndex = myHash(current->key);
                KeyValue<T1, T2> *next = current->next;
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
        table = new KeyValue<T1, T2> *[TABLE_SIZE];

        for (int i = 0; i < TABLE_SIZE; i++)
        {
            table[i] = nullptr;
        }
    }

    bool insert(T1 key, T2 value)
    {
        if (numElements >= TABLE_SIZE)
        {
            resizeTable();
        }

        int index = myHash(key);
        KeyValue<T1, T2> *newNode = new KeyValue<T1, T2>(key, value);

        if (!table[index])
        {
            table[index] = newNode;
        }
        else
        {
            KeyValue<T1, T2> *current = table[index];
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

    bool erase(T1 key)
    {
        int index = myHash(key);

        if (!table[index])
        {
            return false;
        }

        KeyValue<T1, T2> *current = table[index];
        KeyValue<T1, T2> *prev = nullptr;

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

    bool contains(T1 key)
    {
        int index = myHash(key);

        KeyValue<T1, T2> *current = table[index];
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

    T2 &operator[](T1 key)
    {
        if (numElements >= TABLE_SIZE)
        {
            resizeTable();
        }

        int index = myHash(key);
        KeyValue<T1, T2> *current = table[index];

        while (current)
        {
            if (current->key == key)
            {
                return current->value;
            }
            current = current->next;
        }

        KeyValue<T1, T2> *newNode = new KeyValue<T1, T2>(key, T2());
        newNode->next = table[index];
        table[index] = newNode;
        numElements++;
        return table[index]->value;
    }

    void clear()
    {
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValue<T1, T2> *current = table[i];
            while (current)
            {
                KeyValue<T1, T2> *temp = current;
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

    vector<T1> keys()
    {
        vector<T1> result;
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            KeyValue<T1, T2> *current = table[i];
            while (current)
            {
                result.push_back(current->key);
                current = current->next;
            }
        }
        return result;
    }
};

int subarraySumCount(int nums[], int n, int K)
{
    int count = 0;
    int sum = 0;
    UnorderedMap<int, int> sumFrequency; 

    sumFrequency[0] = 1;

    for (int i = 0; i < n; i++)
    {
        sum += nums[i]; 

        if (sumFrequency.contains(sum - K))
        {
            count += sumFrequency[sum - K];
        }

        sumFrequency[sum]++;
    }

    return count;
}

int main()
{
    int N, K;
    cin >> N >> K;
    int nums[N];

    for (int i = 0; i < N; i++)
    {
        cin >> nums[i];
    }

    int result = subarraySumCount(nums, N, K);
    cout << result << endl;

    return 0;
}
