#include <iostream>
#include <vector>
using namespace std;

template <typename KeyType, typename ValueType>
class OrderedMap
{
private:
    struct Node
    {
        KeyType key;
        ValueType value;
        Node *left;
        Node *right;
        int height;

        Node(const KeyType &k, const ValueType &v) : key(k), value(v), left(nullptr), right(nullptr), height(1) {}
    };

    Node *root;
    int numElements; 

    int height(Node *node)
    {
        if (node == nullptr)
        {
            return 0;
        }
        return node->height;
    }

    int balanceFactor(Node *node)
    {
        if (node == nullptr)
        {
            return 0;
        }
        return height(node->left) - height(node->right);
    }

    void updateHeight(Node *node)
    {
        if (node != nullptr)
        {
            node->height = 1 + max(height(node->left), height(node->right));
        }
    }

    Node *rotateRight(Node *y)
    {
        Node *x = y->left;
        Node *rightSubtree = x->right;

        x->right = y;
        y->left = rightSubtree;

        updateHeight(y);
        updateHeight(x);

        return x;
    }

    Node *rotateLeft(Node *x)
    {
        Node *y = x->right;
        Node *leftSubtree = y->left; 

        y->left = x;
        x->right = leftSubtree;

        updateHeight(x);
        updateHeight(y);

        return y;
    }

    Node *insert(Node *node, const KeyType &key, const ValueType &value)
    {
        if (node == nullptr)
        {
            numElements++; 
            return new Node(key, value);
        }

        if (key < node->key)
        {
            node->left = insert(node->left, key, value);
        }
        else if (key > node->key)
        {
            node->right = insert(node->right, key, value);
        }
        else
        {
            return node;
        }

        updateHeight(node);

        int balance = balanceFactor(node);

        if (balance > 1)
        {
            if (key < node->left->key)
            {
                return rotateRight(node);
            }
            else
            {
                node->left = rotateLeft(node->left);
                return rotateRight(node);
            }
        }

        if (balance < -1)
        {
            if (key > node->right->key)
            {
                return rotateLeft(node);
            }
            else
            {
                node->right = rotateRight(node->right);
                return rotateLeft(node);
            }
        }

        return node;
    }

    Node *findMin(Node *node)
    {
        while (node->left != nullptr)
            node = node->left;
        return node;
    }

    Node *erase(Node *node, const KeyType &key)
    {
        if (node == nullptr)
        {
            return node;
        }

        if (key < node->key)
        {
            node->left = erase(node->left, key);
        }
        else if (key > node->key)
        {
            node->right = erase(node->right, key);
        }
        else
        {
            if (node->left == nullptr || node->right == nullptr)
            {
                Node *temp = node->left ? node->left : node->right;

                if (temp == nullptr)
                {
                    temp = node;
                    node = nullptr;
                }
                else
                {
                    *node = *temp;
                }

                delete temp;
                numElements--; 
            }
            else
            {
                Node *temp = findMin(node->right);
                node->key = temp->key;
                node->value = temp->value;
                node->right = erase(node->right, temp->key);
            }
        }

        if (node == nullptr){
            return node;
        }

        updateHeight(node);

        int balance = balanceFactor(node);

        if (balance > 1)
        {
            if (balanceFactor(node->left) >= 0)
            {
                return rotateRight(node);
            }
            else
            {
                node->left = rotateLeft(node->left);
                return rotateRight(node);
            }
        }

        if (balance < -1)
        {
            if (balanceFactor(node->right) <= 0)
            {
                return rotateLeft(node);
            }
            else
            {
                node->right = rotateRight(node->right);
                return rotateLeft(node);
            }
        }

        return node;
    }

    bool contains(Node *node, const KeyType &key)
    {
        if (node == nullptr)
        {
            return false;
        }

        if (key == node->key)
        {
            return true;
        }
        else if (key < node->key)
        {
            return contains(node->left, key);
        }
        else
        {
            return contains(node->right, key);
        }
    }

    void inorderTraversal(Node *node, vector<KeyType> &result)
    {
        if (node == nullptr)
        {
            return;
        }

        inorderTraversal(node->left, result);
        result.push_back(node->key);
        inorderTraversal(node->right, result);
    }

    void clear(Node *node)
    {
        if (node == nullptr)
        {
            return;
        }

        clear(node->left);
        clear(node->right);
        delete node;
    }

public:
    OrderedMap() : root(nullptr), numElements(0) {}

    bool empty()
    {
        return numElements == 0;
    }

    int size()
    {
        return numElements;
    }

    bool contains(const KeyType &key)
    {
        return contains(root, key);
    }

    bool insert(const KeyType &key, const ValueType &value)
    {
        int sizeBeforeInsertion = size();
        root = insert(root, key, value);
        int sizeAfterInsertion = size();

        return sizeAfterInsertion > sizeBeforeInsertion;
    }

    bool erase(const KeyType &key)
    {
        int sizeBeforeErase = size();
        root = erase(root, key);
        int sizeAfterErase = size();

        return sizeAfterErase < sizeBeforeErase;
    }

    ValueType &operator[](const KeyType &key)
    {
        Node *node = root;
        while (node != nullptr)
        {
            if (key == node->key)
            {
                return node->value;
            }
            else if (key < node->key)
            {
                if (node->left == nullptr)
                {
                    insert(key, ValueType());
                    return node->left->value;
                }
                node = node->left;
            }
            else
            {
                if (node->right == nullptr)
                {
                    insert(key, ValueType());
                    return node->right->value;
                }
                node = node->right;
            }
        }
        insert(key, ValueType());
        return root->value;
    }

    void clear()
    {
        clear(root);
        root = nullptr;
        numElements = 0; 
    }

    vector<KeyType> keys()
    {
        vector<KeyType> result;
        inorderTraversal(root, result);
        return result;
    }

    pair<bool, KeyType> lower_bound(const KeyType &key)
    {
        Node *node = root;
        KeyType lb;
        bool found = false;

        while (node != nullptr)
        {
            if (key <= node->key)
            {
                lb = node->key;
                node = node->left;
                found = true;
            }
            else
            {
                node = node->right;
            }
        }

        if (found)
        {
            return make_pair(true, lb);
        }
        else
        {
            return make_pair(false, KeyType()); 
        }
    }

    pair<bool, KeyType> upper_bound(const KeyType &key)
    {
        Node *node = root;
        KeyType ub;
        bool found = false;

        while (node != nullptr)
        {
            if (key < node->key)
            {
                ub = node->key;
                node = node->left;
                found = true;
            }
            else
            {
                node = node->right;
            }
        }

        if (found)
        {
            return make_pair(true, ub);
        }
        else
        {
            return make_pair(false, KeyType()); 
        }
    }
};

int main()
{
    OrderedMap<int, string> myMap;
    // int(KeyType) key; string(ValueType) value; vector<int(KeyType)> keys = myMap.keys();

    while (true)
    {
        // cout << "Menu:\n";
        // cout << "0. Exit\n";
        // cout << "1. Check if empty\n";
        // cout << "2. Get size\n";
        // cout << "3. Check if key exists\n";
        // cout << "4. Insert key-value pair\n";
        // cout << "5. Erase key\n";
        // cout << "6. Access key with default value\n";
        // cout << "7. Clear map\n";
        // cout << "8. Get keys in sorted order\n";
        // cout << "9. Lower bound\n";
        // cout << "10. Upper bound\n";
        // cout << "Enter your choice: ";

        int choice;
        cin >> choice;

        int key;
        string value;
        vector<int> keys = myMap.keys();

        switch (choice)
        {
        case 1:
            cout << (myMap.empty() ? "true" : "false") << endl;
            break;
        case 2:
            cout << myMap.size() << endl;
            break;
        case 3:
        {
            // int key;
            // cout << "Enter key to check: ";
            cin >> key;
            cout << (myMap.contains(key) ? "true" : "false") << endl;
            break;
        }
        case 4:
        {
            // cout << "Enter key and value to insert: ";
            cin >> key >> value;
            cout << (myMap.insert(key, value) ? "true" : "false") << endl;
            break;
        }
        case 5:
        {
            // int key;
            // cout << "Enter key to erase: ";
            cin >> key;
            cout << (myMap.erase(key) ? "true" : "false") << endl;
            break;
        }
        case 6:
        {
            // int key;
            // cout << "Enter key to access: ";
            cin >> key;
            cout << myMap[key] << endl;
            break;
        }
        case 7:
            myMap.clear();
            break;
        case 8:
        {
            for (int key : keys)
            {
                cout << key << endl;
            }
            break;
        }
        case 9:
        {
            // int key;
            // cout << "Enter key for lower bound: ";
            cin >> key;
            auto lb = myMap.lower_bound(key);
            if (lb.first)
            {
                cout << "true\n" << lb.second << endl;
            }
            else
            {
                cout << "false" << endl;
            }
            break;
        }
        case 10:
        {
            // int key;
            // cout << "Enter key for upper bound: ";
            cin >> key;
            auto ub = myMap.upper_bound(key);
            if (ub.first)
            {
                cout << "true\n" << ub.second << endl;
            }
            else
            {
                cout << "false" << endl;
            }
            break;
        }
        case 0:
            return 0;
        default:
            cout << "Invalid choice, please try again." << endl;
        }
    }

    return 0;
}
