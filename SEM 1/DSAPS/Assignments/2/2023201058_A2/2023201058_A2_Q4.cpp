#include <iostream>
#include <ctime>
#include <climits>
using namespace std;

// Class for Skip List Node
template <typename T, typename Comparator = less<T>>
struct SkipListNode
{
    T value;
    SkipListNode **next;

    SkipListNode(T val, int level) : value(val)
    {
        next = new SkipListNode *[level];
        for (int i = 0; i < level; i++)
        {
            next[i] = nullptr;
        }
    }

    ~SkipListNode()
    {
        delete[] next;
    }

    // Custom comparator function for class data types
    static Comparator comp; 
};

template <typename T, typename Comparator>
Comparator SkipListNode<T, Comparator>::comp;

template <typename T, typename Comparator = less<T>>
class SkipList
{
public:
    SkipList();
    ~SkipList();

    void insert(const T &value);
    void remove(const T &value);
    bool search(const T &value);
    int count_occurrence(const T &value);
    T lower_bound(const T &value);
    T upper_bound(const T &value);
    T closest_element(const T &value);

    void print();

private:
    int max_level;
    SkipListNode<T, Comparator> *head;

    int random_level();
    SkipListNode<T, Comparator> *create_node(const T &value, int level);
};

template <typename T, typename Comparator>
SkipList<T, Comparator>::SkipList() : max_level(1)
{
    head = create_node(T(), max_level);
}

template <typename T, typename Comparator>
SkipList<T, Comparator>::~SkipList()
{
    SkipListNode<T, Comparator> *current = head;
    while (current)
    {
        SkipListNode<T, Comparator> *next = current->next[0];
        delete current;
        current = next;
    }
}

template <typename T, typename Comparator>
int SkipList<T, Comparator>::random_level()
{
    int level = 1;
    while ((rand() % 2) && (level < max_level + 1))
        level++;
    return level;
}

template <typename T, typename Comparator>
SkipListNode<T, Comparator> *SkipList<T, Comparator>::create_node(const T &value, int level)
{
    return new SkipListNode<T, Comparator>(value, level);
}

template <typename T, typename Comparator>
void SkipList<T, Comparator>::insert(const T &value)
{
    int level = random_level();
    if (level > max_level)
    {
        max_level = level;
        head->next = (SkipListNode<T, Comparator> **)realloc(head->next, max_level * sizeof(SkipListNode<T, Comparator> *));
        for (int i = max_level - 1; i >= level; i--)
        {
            head->next[i] = nullptr;
        }
    }

    SkipListNode<T, Comparator> *newNode = create_node(value, level);

    SkipListNode<T, Comparator> *current = head;
    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (i < level)
        {
            newNode->next[i] = current->next[i];
            current->next[i] = newNode;
        }
    }
}

template <typename T, typename Comparator>
void SkipList<T, Comparator>::remove(const T &value)
{
    SkipListNode<T, Comparator> *current = head;
    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (current->next[i] && current->next[i]->value == value)
        {
            SkipListNode<T, Comparator> *toRemove = current->next[i];
            current->next[i] = toRemove->next[i];
            delete toRemove;
        }
    }
}

template <typename T, typename Comparator>
bool SkipList<T, Comparator>::search(const T &value)
{
    SkipListNode<T, Comparator> *current = head;
    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (current->next[i] && current->next[i]->value == value)
            return true;
    }
    return false;
}

template <typename T, typename Comparator>
int SkipList<T, Comparator>::count_occurrence(const T &value)
{
    int count = 0;
    SkipListNode<T, Comparator> *current = head->next[0]; // Start from the first element

    while (current)
    {
        if (current->value == value)
        {
            count++;
        }
        current = current->next[0]; // Move to the next element
    }

    return count;
}

template <typename T, typename Comparator>
T SkipList<T, Comparator>::lower_bound(const T &value)
{
    SkipListNode<T, Comparator> *current = head;
    T result = T();

    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (current->next[i] && !current->comp(current->next[i]->value, value))
        {
            result = current->next[i]->value;
        }
    }

    return result;
}

template <typename T, typename Comparator>
T SkipList<T, Comparator>::upper_bound(const T &value)
{
    SkipListNode<T, Comparator> *current = head;
    T result = T();

    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (current->next[i] && !current->comp(current->next[i]->value, value))
        {
            // Check for duplicates and find the first occurrence
            while (current->next[i] && current->next[i]->value == value)
                current = current->next[i];

            if (current->next[i])
            {
                result = current->next[i]->value;
            }
        }
    }

    return result;
}

template <typename T, typename Comparator>
T SkipList<T, Comparator>::closest_element(const T &value)
{
    SkipListNode<T, Comparator> *current = head;
    T closest = head->value;

    for (int i = max_level - 1; i >= 0; i--)
    {
        while (current->next[i] && current->comp(current->next[i]->value, value))
            current = current->next[i];

        if (current->next[i])
        {
            T next_value = current->next[i]->value;

            // Check if next_value is closer to value than the current closest
            if (!current->comp(next_value, value) && (current->comp(value, closest) || closest == head->value))
            {
                closest = next_value;
            }
        }
    }

    return closest;
}

template <typename T, typename Comparator>
void SkipList<T, Comparator>::print()
{
    for (int i = max_level - 1; i >= 0; i--)
    {
        SkipListNode<T, Comparator> *current = head->next[i];
        cout << "Level " << i << ": ";
        while (current)
        {
            cout << current->value << " ";
            current = current->next[i];
        }
        cout << endl;
    }
}

// int main()
// {
//     // Example usage of SkipList
//     SkipList<int> skipList;
//     skipList.insert(1);
//     skipList.insert(1);
//     skipList.insert(2);
//     skipList.insert(2);
//     skipList.insert(2);
//     skipList.insert(3);
//     skipList.insert(5);
//     skipList.insert(2);
//     // skipList.insert(8);
//     skipList.insert(70);

//     skipList.print();

//     cout << "Count of 2: " << skipList.count_occurrence(2) << endl;
//     cout << "Count of 734: " << skipList.count_occurrence(734) << endl;
//     cout << "Lower bound of 4: " << skipList.lower_bound(4) << endl;
//     cout << "Lower bound of 3: " << skipList.lower_bound(3) << endl;
//     cout << "Upper bound of 2: " << skipList.upper_bound(2) << endl;
//     cout << "Upper bound of 7: " << skipList.upper_bound(7) << endl;
//     cout << "Closest element to 2: " << skipList.closest_element(2) << endl;
//     cout << "Closest element to 3: " << skipList.closest_element(3) << endl;
//     cout << "Closest element to 4: " << skipList.closest_element(4) << endl;
//     cout << "Closest element to -1472: " << skipList.closest_element(-1472) << endl;
//     cout << "Closest element to 6: " << skipList.closest_element(6) << endl;

//     return 0;
// }
int main()
{
    int N;
    cin >> N;

    // create skiplist
    SkipList<int> skipList;

    int data, op;

    while (N > 0)
    {
        cin >> op;
        cin >> data;
        switch (op)
        {
        case 1: // insert
            skipList.insert(data);
            break;
        case 2: // delete
            skipList.remove(data);
            break;
        case 3: // search
            // print result
            // cout << (skipList.search(data) ? "True" : "False");
            cout << skipList.search(data);
            cout << "\n";
            break;
        case 4: // count
            // print result
            cout << skipList.count_occurrence(data);
            cout << "\n";
            break;
        case 5: // lower bound
            // print result
            cout << skipList.lower_bound(data);
            cout << "\n";
            break;
        case 6: // upper bound
            // print result
            cout << skipList.upper_bound(data);
            cout << "\n";
            break;
        case 7: // closest val
            // print result
            cout << skipList.closest_element(data);
            cout << "\n";
            break;
        }
        --N;
    }

    return 0;
}
