using namespace std;
#include <iostream>
#include <vector>

template <typename T1, typename T2>
class OrderedMap
{
private:
    struct Node
    {
        T1 key;
        T2 value;
        Node *left;
        Node *right;
        int height;

        Node(const T1 &k, const T2 &v) : key(k), value(v), left(nullptr), right(nullptr), height(1) {}
    };

    Node *root;
    int numElements; 

    int height(Node *node)
    {
        if (node == nullptr)
            return 0;
        return node->height;
    }

    int balanceFactor(Node *node)
    {
        if (node == nullptr)
            return 0;
        return height(node->left) - height(node->right);
    }

    void updateHeight(Node *node)
    {
        if (node != nullptr)
            node->height = 1 + max(height(node->left), height(node->right));
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

    Node *insert(Node *node, const T1 &key, const T2 &value)
    {
        if (node == nullptr)
        {
            numElements++;
            return new Node(key, value);
        }

        if (key < node->key)
            node->left = insert(node->left, key, value);
        else if (key > node->key)
            node->right = insert(node->right, key, value);
        else
            return node;

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

    Node *erase(Node *node, const T1 &key)
    {
        if (node == nullptr)
            return node;

        if (key < node->key)
            node->left = erase(node->left, key);
        else if (key > node->key)
            node->right = erase(node->right, key);
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
                    *node = *temp;

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
            // cout << numElements << endl;
        }

        if (node == nullptr)
            return node;

        updateHeight(node);

        int balance = balanceFactor(node);

        if (balance > 1)
        {
            if (balanceFactor(node->left) >= 0)
                return rotateRight(node);
            else
            {
                node->left = rotateLeft(node->left);
                return rotateRight(node);
            }
        }

        if (balance < -1)
        {
            if (balanceFactor(node->right) <= 0)
                return rotateLeft(node);
            else
            {
                node->right = rotateRight(node->right);
                return rotateLeft(node);
            }
        }

        return node;
    }

    bool contains(Node *node, const T1 &key)
    {
        if (node == nullptr)
            return false;

        if (key == node->key)
            return true;
        else if (key < node->key)
            return contains(node->left, key);
        else
            return contains(node->right, key);
    }

    void inorderTraversal(Node *node, vector<T1> &result)
    {
        if (node == nullptr)
            return;

        inorderTraversal(node->left, result);
        result.push_back(node->key);
        inorderTraversal(node->right, result);
    }

    void clear(Node *node)
    {
        if (node == nullptr)
            return;

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

    bool contains(const T1 &key)
    {
        return contains(root, key);
    }

    bool insert(const T1 &key, const T2 &value)
    {
        int sizeBeforeInsertion = size();
        root = insert(root, key, value);
        int sizeAfterInsertion = size();

        return sizeAfterInsertion > sizeBeforeInsertion;
    }

    bool erase(const T1 &key)
    {
        int sizeBeforeErase = size();
        root = erase(root, key);
        int sizeAfterErase = size();

        return sizeAfterErase < sizeBeforeErase;
    }

    T2 &operator[](const T1 &key)
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
                    insert(key, T2());
                    return node->left->value;
                }
                node = node->left;
            }
            else
            {
                if (node->right == nullptr)
                {
                    insert(key, T2());
                    return node->right->value;
                }
                node = node->right;
            }
        }
        insert(key, T2());
        return root->value;
    }

    void clear()
    {
        clear(root);
        root = nullptr;
        numElements = 0; 
    }

    vector<T1> keys()
    {
        vector<T1> result;
        inorderTraversal(root, result);
        return result;
    }

    pair<bool, T1> lower_bound(const T1 &key)
    {
        Node *node = root;
        T1 lb;
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
            return make_pair(false, T1()); 
        }
    }

    pair<bool, T1> upper_bound(const T1 &key)
    {
        Node *node = root;
        T1 ub;
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
            return make_pair(false, T1()); 
        }
    }
};

int main()
{
    int N;
    cin >> N;
    OrderedMap<pair<int, int>, int> tokenCount;

    for (int i = 0; i < N; i++)
    {
        int top_view, bottom_view;
        cin >> top_view >> bottom_view;
        pair<int, int> token = make_pair(top_view, bottom_view);

        
        if (top_view > bottom_view)
        {
            swap(top_view, bottom_view);
            token = make_pair(top_view, bottom_view);
        }

        tokenCount[token]++;
    }

    long long equivalentPairs = 0;
    vector<pair<int, int>> tokens = tokenCount.keys();

    for (const auto &token : tokens)
    {
        int count = tokenCount[token];
        equivalentPairs += (count * 1LL * (count - 1)) / 2; 
    }

    cout << equivalentPairs << endl;

    return 0;
}
