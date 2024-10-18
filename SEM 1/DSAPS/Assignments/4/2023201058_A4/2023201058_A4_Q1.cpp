#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

template <typename T>
class treap
{
public:
    struct Node
    {
        T data;
        int priority;
        int size;
        Node *left;
        Node *right;

        Node(const T &val) : data(val), priority(rand()), size(1), left(nullptr), right(nullptr) {}
    };

    Node *root;

    int getSize(Node *node)
    {
        return node ? node->size : 0;
    }

    void updateSize(Node *node)
    {
        if (node)
        {
            node->size = 1 + getSize(node->left) + getSize(node->right);
        }
    }

    Node *mergeNodes(Node *leftNode, Node *rightNode)
    {
        if (!leftNode)
            return rightNode;
        if (!rightNode)
            return leftNode;

        if (leftNode->priority > rightNode->priority)
        {
            leftNode->right = mergeNodes(leftNode->right, rightNode);
            updateSize(leftNode);
            return leftNode;
        }
        else
        {
            rightNode->left = mergeNodes(leftNode, rightNode->left);
            updateSize(rightNode);
            return rightNode;
        }
    }

    pair<Node *, Node *> split(Node *node, int index)
    {
        if (!node)
            return {nullptr, nullptr};

        int leftSize = getSize(node->left);
        if (leftSize >= index)
        {
            auto leftSplit = split(node->left, index);
            node->left = leftSplit.second;
            updateSize(node);
            return {leftSplit.first, node};
        }
        else
        {
            auto rightSplit = split(node->right, index - leftSize - 1);
            node->right = rightSplit.first;
            updateSize(node);
            return {node, rightSplit.second};
        }
    }

    bool erase(Node *&node, int index)
    {
        if (!node)
            return false;

        int leftSize = getSize(node->left);
        if (leftSize == index)
        {
            Node *temp = node;
            node = mergeNodes(node->left, node->right);
            delete temp;
            updateSize(node);
            return true;
        }
        else if (leftSize > index)
        {
            return erase(node->left, index);
        }
        else
        {
            return erase(node->right, index - leftSize - 1);
        }
    }

    int indexOf(Node *node, const T &val, int index = 0)
    {
        if (!node)
            return -1;

        if (val < node->data)
        {
            return indexOf(node->left, val, index);
        }
        else if (val > node->data)
        {
            return indexOf(node->right, val, index + 1 + getSize(node->left));
        }
        else
        {
            return index + getSize(node->left);
        }
    }

    T atIndex(Node *node, int index)
    {
        if (!node)
        {

            return T();
        }

        int leftSize = getSize(node->left);
        if (leftSize == index)
        {
            return node->data;
        }
        else if (leftSize > index)
        {
            return atIndex(node->left, index);
        }
        else
        {
            return atIndex(node->right, index - leftSize - 1);
        }
    }

    int lowerBound(Node *node, const T &val)
    {
        if (!node)
            return 0;

        if (val <= node->data)
        {
            return lowerBound(node->left, val);
        }
        else
        {
            return 1 + getSize(node->left) + lowerBound(node->right, val);
        }
    }

    int upperBound(Node *node, const T &val)
    {
        if (!node)
            return 0;

        if (val < node->data)
        {
            return upperBound(node->left, val);
        }
        else
        {
            return 1 + getSize(node->left) + upperBound(node->right, val);
        }
    }

    int count(Node *node, const T &val)
    {
        if (!node)
            return 0;

        if (val < node->data)
        {
            return count(node->left, val);
        }
        else if (val > node->data)
        {
            return count(node->right, val);
        }
        else
        {
            return 1 + count(node->left, val) + count(node->right, val);
        }
    }

    void toVector(Node *node, vector<T> &result)
    {
        if (node)
        {
            toVector(node->left, result);
            result.push_back(node->data);
            toVector(node->right, result);
        }
    }

    treap() : root(nullptr)
    {

        srand(time(0));
    }

    bool empty()
    {
        return !root;
    }

    int size()
    {
        return getSize(root);
    }

    void clear()
    {
        while (root)
        {
            erase(root, 0);
        }
    }

    int insert(const T &val)
    {
        int index = lowerBound(root, val);
        auto splitNodes = split(root, index);
        Node *newNode = new Node(val);
        root = mergeNodes(mergeNodes(splitNodes.first, newNode), splitNodes.second);
        return index;
    }

    bool erase(int index)
    {
        return erase(index, index);
    }

    int indexOf(const T &val)
    {
        return indexOf(root, val);
    }

    T atIndex(int index)
    {
        return atIndex(root, index);
    }

    treap<T> *merge(treap<T> *t2)
    {
        if (!t2)
            return this;
        root = mergeNodes(root, t2->root);
        t2->root = nullptr;
        return this;
    }

    pair<treap<T> *, treap<T> *> split(int index)
    {
        auto splitNodes = split(root, index);
        treap<T> *t1 = new treap<T>;
        treap<T> *t2 = new treap<T>;
        t1->root = splitNodes.first;
        t2->root = splitNodes.second;
        root = nullptr;
        return {t1, t2};
    }

    bool erase(int first, int last)
    {
        int size = getSize(root);
        if (first < 0 || last >= size)
            return false;

        auto split1 = split(root, first);
        auto split2 = split(split1.second, last - first + 1);
        root = mergeNodes(split1.first, split2.second);

        return true;
    }

    treap<T> *slice(int first, int last)
    {
        int size = getSize(root);
        if (first - last > 0 || first < 0 || last >= size)
            return nullptr;

        auto split1 = split(root, first);
        auto split2 = split(split1.second, last - first + 1);

        treap<T> *result = new treap<T>;
        result->root = split2.first;
        root = mergeNodes(split1.first, split2.second);

        return result;
    }

    int lower_bound(const T &val)
    {
        return lowerBound(root, val);
    }

    int upper_bound(const T &val)
    {
        return upperBound(root, val);
    }

    int count(const T &val)
    {
        return count(root, val);
    }

    vector<T> to_array()
    {
        vector<T> result;
        toVector(root, result);
        return result;
    }
};

int main()
{
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    treap<int> t;

    while (true)
    {
        /*cout << "1. bool empty()  ";
        cout << "2. int size()  ";
        cout << "3. void clear()  ";
        cout << "4. int insert(T val)  ";
        cout << "5. bool erase(int index)  ";
        cout << "6. int indexOf(T val)  ";
        cout << "7. T atIndex(int index)  ";
        cout << "8. treap<T>* merge(treap<T> *t2)  ";
        cout << "9. pair<treap<T>, treap<T>> split(int index)  ";
        cout << "10. bool erase(int first, int last)  ";
        cout << "11. treap<T>* slice(int first, int last)  ";
        cout << "12. int lower_bound(T val)  ";
        cout << "13. int upper_bound(T val)  ";
        cout << "14. int count(T val)  ";
        cout << "15. vector<T> to_array()  ";
        cout << "0. Exit  ";
        cout << "\nEnter your choice: ";*/

        int choice;
        cin >> choice;

        if (choice == 0)
        {
            // Exit the program
            break;
        }

        switch (choice)
        {
        case 1:
        {
            bool is_empty = t.empty();
            cout << boolalpha << is_empty << endl;
            // cout << (is_empty ? "The treap is empty" : "The treap is not empty") << endl;
            break;
        }
        case 2:
        {
            int size = t.size();
            cout << size << endl;
            // cout << "Size of the treap: " << size << endl;
            break;
        }
        case 3:
        {
            t.clear();
            // cout << "Cleared the treap" << endl;
            break;
        }
        case 4:
        {
            int val;
            // cout << "Enter the value to insert: ";
            cin >> val;
            int index = t.insert(val);
            cout << index << endl;
            // cout << "Inserted at index: " << index << endl;
            break;
        }
        case 5:
        {
            int index;
            // cout << "Enter the index to delete: ";
            cin >> index;
            bool success = t.erase(index);
            cout << boolalpha << success << endl;
            // cout << (success ? "Deleted successfully" : "Deletion failed") << endl;
            break;
        }
        case 6:
        {
            int val;
            // cout << "Enter the value to find: ";
            cin >> val;
            int index = t.indexOf(val);
            cout << index << endl;
            // cout << "Index of value: " << index << endl;
            break;
        }
        case 7:
        {
            int index;
            // cout << "Enter the index to find: ";
            cin >> index;
            int value = t.atIndex(index);
            cout << value << endl;
            // cout << "Value at index: " << value << endl;
            break;
        }
        case 8:
        {
            treap<int> t2;
            // Populate t2 as needed
            int val;
            while (true)
            {
                cin >> val;
                if (val == -1)
                    break;
                t2.insert(val);
            }
            // Merge with another treap
            treap<int> *mergedTreap = t.merge(&t2);
            // cout << "Merged successfully" << endl;
            break;
        }
        case 9:
        {
            int index;
            // cout << "Enter the index to split: ";
            cin >> index;
            auto splits = t.split(index);
            // Handle splits as needed
            treap<int> *leftTreap = splits.first;
            treap<int> *rightTreap = splits.second;
            // cout << "Split the treap" << endl;
            break;
        }
        case 10:
        {
            int first, last;
            // cout << "Enter the first and last indices to delete (inclusive): ";
            cin >> first >> last;
            bool success = t.erase(first, last);
            cout << boolalpha << success << endl;
            // cout << (success ? "Deleted successfully" : "Deletion failed") << endl;
            break;
        }
        case 11:
        {
            int first, last;
            // cout << "Enter the first and last indices for the slice (inclusive): ";
            cin >> first >> last;
            treap<int> *sliced = t.slice(first, last);
            if (sliced)
            {
                // Handle the slice treap
                t = *sliced;
                // cout << "Created a slice" << endl;
            }
            else
            {
                // cout << "Slice operation failed" << endl;
            }
            break;
        }
        case 12:
        {
            int val;
            // cout << "Enter the value to find the lower bound: ";
            cin >> val;
            int lb = t.lower_bound(val);
            cout << lb << endl;
            // cout << "Lower bound: " << lb << endl;
            break;
        }
        case 13:
        {
            int val;
            // cout << "Enter the value to find the upper bound: ";
            cin >> val;
            int ub = t.upper_bound(val);
            cout << ub << endl;
            // cout << "Upper bound: " << ub << endl;
            break;
        }
        case 14:
        {
            int val;
            // cout << "Enter the value to count: ";
            cin >> val;
            int count = t.count(val);
            cout << count << endl;
            // cout << "Count: " << count << endl;
            break;
        }
        case 15:
        {
            vector<int> arr = t.to_array();
            // cout << "Treap elements in sorted order: ";
            for (int val : arr)
            {
                cout << val << " ";
            }
            cout << endl;
            break;
        }
        default:
            cout << "Invalid choice. Please try again." << endl;
        }
    }

    return 0;
}
