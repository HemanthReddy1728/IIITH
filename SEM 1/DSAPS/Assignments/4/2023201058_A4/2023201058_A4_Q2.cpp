#include <iostream>
#include <string>
#include <functional>
using namespace std;

class rope
{
public:
    struct Node
    {
        string str;
        int length;
        Node *left;
        Node *right;

        Node(const string &s) : str(s), length(s.length()), left(nullptr), right(nullptr) {}
    };

    Node *root;

    int getLength(Node *node)
    {
        return node ? node->length : 0;
    }

    void updateLength(Node *node)
    {
        if (node)
        {
            node->length = getLength(node->left) + getLength(node->right) + int(node->str.length());
        }
    }

    Node *concat(Node *r1, Node *r2)
    {
        if (!r1)
            return r2;
        if (!r2)
            return r1;

        Node *newRoot = new Node("");
        newRoot->left = r1;
        newRoot->right = r2;
        updateLength(newRoot);

        return newRoot;
    }

    pair<Node *, Node *> split(Node *node, int index)
    {
        if (!node)
            return {nullptr, nullptr};

        if (index <= getLength(node->left))
        {
            auto [left, right] = split(node->left, index);
            node->left = right;
            updateLength(node);
            return {left, node};
        }
        else if (index >= getLength(node->left) + int(node->str.length()))
        {
            auto [left, right] = split(node->right, index - getLength(node->left) - int(node->str.length()));
            node->right = left;
            updateLength(node);
            return {node, right};
        }
        else
        {
            int splitIndex = index - getLength(node->left);
            Node *left = new Node(node->str.substr(0, splitIndex));
            Node *right = new Node(node->str.substr(splitIndex));

            left->left = node->left;
            right->right = node->right;
            updateLength(left);
            updateLength(right);

            return {left, right};
        }
    }

    void clear(Node *node)
    {
        if (node)
        {
            clear(node->left);
            clear(node->right);

            if (node->left)
                node->left->right = nullptr;
            if (node->right)
                node->right->left = nullptr;

            delete node;
        }
    }

    void inorderTraversal(Node *node, function<void(Node *)> &visit)
    {
        if (node)
        {
            inorderTraversal(node->left, visit);
            visit(node);
            inorderTraversal(node->right, visit);
        }
    }

    Node *copyNode(Node *node)
    {
        if (!node)
            return nullptr;

        Node *newNode = new Node(node->str);
        newNode->left = copyNode(node->left);
        newNode->right = copyNode(node->right);
        updateLength(newNode);

        return newNode;
    }

    rope(const string &s) : root(new Node(s)) {}

    rope(const rope &other) : root(nullptr)
    {
        if (other.root)
            root = copyNode(other.root);
    }

    rope &operator=(const rope &other)
    {
        if (this != &other)
        {
            clear(root);
            root = nullptr;
            if (other.root)
                root = copyNode(other.root);
        }
        return *this;
    }

    // ~rope()
    // {
    //     clear(root);
    // }

    bool empty()
    {
        return root == nullptr;
    }

    int size()
    {
        return getLength(root);
    }

    void clear()
    {
        clear(root);
        root = nullptr;
    }

    bool insert(int i, const string &s)
    {
        if (i < 0 || i > size())
        {
            return false;
        }

        auto [left, right] = split(root, i);
        root = concat(concat(left, new Node(s)), right);

        return true;
    }

    bool erase(int first, int last)
    {
        if (first < 0 || last >= size() || first > last)
        {
            return false;
        }

        auto [left, right] = split(root, first);
        auto [mid, temp] = split(right, last - first + 1);
        clear(mid);
        root = concat(left, temp);

        return true;
    }

    char charAt(int index)
    {
        if (index < 0 || index >= size())
        {
            return '\0';
        }

        Node *current = root;
        while (current)
        {
            if (index < getLength(current->left))
            {
                current = current->left;
            }
            else if (index >= getLength(current->left) + int(current->str.length()))
            {
                index -= getLength(current->left) + int(current->str.length());
                current = current->right;
            }
            else
            {
                return current->str[index - getLength(current->left)];
            }
        }

        return '\0';
    }

    rope *subrope(int first, int last)
    {
        if (first < 0 || last >= size() || first > last)
        {
            return nullptr;
        }

        auto [left, right] = split(root, first);
        auto [mid, temp] = split(right, last - first + 1);

        rope *result = new rope(mid->str);

        root = concat(left, right);

        return result;
    }

    rope *concat(rope *r2)
    {
        root = concat(root, r2->root);
        return this;
    }

    rope *push_back(const string &s)
    {
        auto [left, right] = split(root, size());
        // root = concat(root, new Node(s));
        root = concat(concat(left, new Node(s)), right);
        return this;
    }

    string to_string()
    {
        if (!root)
        {
            return "";
        }

        string result;
        function<void(Node *)> visit = [&](Node *node)
        {
            result += node->str;
        };

        inorderTraversal(root, visit);

        return result;
    }

    pair<rope *, rope *> split(int index)
    {
        if (index < 0 || index >= size())
        {
            return {};
        }
        auto [left, right] = split(root, index);
        root = left;
        return {new rope(left->str), new rope(right->str)};
    }
};

int main()
{
    // ios_base::sync_with_stdio(false); cin.tie(NULL);
    rope *r1 = nullptr;
    int op, choice;
    cin >> op;
    while (op--)
    {
        /*cout << "1. void rope(string s)  ";
        cout << "2. bool empty()  ";
        cout << "3. int size()  ";
        cout << "4. void clear()  ";
        cout << "5. bool insert(int i, string s)  ";
        cout << "6. bool erase(int first, int last)  ";
        cout << "7. char charAt(int index)  ";
        cout << "8. rope* subrope(int first, int last)  ";
        cout << "9. rope* concat(rope* r2)  ";
        cout << "10. rope* push_back(string s)  ";
        cout << "11. string to_string()  ";
        cout << "12. pair<rope*, rope*> split(int index)  ";
        cout << "0. Exit  ";

        cout << "\nEnter your choice: ";*/
        cin >> choice;

        switch (choice)
        {
        case 1:
        {
            string s;
            // cout << "Enter the string for the rope: ";
            cin >> s;
            r1 = new rope(s);
            break;
        }
        case 2:
        {
            cout << r1->empty() << '\n';
            break;
        }
        case 3:
        {
            cout << r1->size() << '\n';
            break;
        }
        case 4:
        {
            r1->clear();
            break;
        }
        case 5:
        {
            int i;
            string s;
            // cout << "Enter the index and string to insert: ";
            cin >> i >> s;
            cout << r1->insert(i, s) << '\n';
            break;
        }
        case 6:
        {
            int first, last;
            // cout << "Enter the range to erase (first and last indices): ";
            cin >> first >> last;
            cout << r1->erase(first, last) << '\n';
            break;
        }
        case 7:
        {
            int index;
            // cout << "Enter the index to get char: ";
            cin >> index;
            cout << r1->charAt(index) << '\n';
            break;
        }
        case 8:
        {
            int first, last;
            // cout << "Enter the range for subrope (first and last indices): ";
            cin >> first >> last;
            rope *sub = r1->subrope(first, last);
            if (sub)
            {
                delete r1;
                r1 = sub;
                // cout << r1->to_string() << '\n';
            }
            else
            {
                // cout << "Invalid range  ";
            }
            break;
        }
        case 9:
        {
            rope *r2;
            string s;
            // cout << "Enter the string for the second rope: ";
            cin >> s;
            r2 = new rope(s);
            r1->concat(r2);
            // rope *cat = r1->concat(r2);
            // delete r1;
            // r1 = cat;
            // cout << r1->to_string() << '\n';
            delete r2;
            break;
        }
        case 10:
        {
            string s;
            // cout << "Enter the string to push back: ";
            cin >> s;
            r1->push_back(s);
            // rope *pb = r1->push_back(s);
            // delete r1;
            // r1 = pb;
            // cout << r1->to_string() << '\n';
            break;
        }
        case 11:
        {
            cout << r1->to_string() << '\n';
            break;
        }
        case 12:
        {
            int index;
            // cout << "Enter the index to split the rope: ";
            cin >> index;
            auto [r3, r4] = r1->split(index);
            // cout << "First rope: " << r3->to_string() << '\n';
            // cout << "Second rope: " << r4->to_string() << '\n';
            delete r3;
            delete r4;
            break;
        }
        case 0:
        {
            delete r1;
            // cout << "Exiting...  ";
            break;
        }
        default:
            cout << "Invalid choice  ";
        }
    }

    return 0;
}
