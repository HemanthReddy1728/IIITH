#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

// Structure to represent a non-zero element in the matrix
template <typename T>
struct Node
{
    int row, col;
    T value;
    Node *next;
};

// Class to represent a sparse matrix using a linked list
template <typename T>
class SparseMatrix
{
private:
    int rows, cols;
    Node<T> *head;
    Node<T> *tail;

public:
    SparseMatrix(int rows, int cols) : rows(rows), cols(cols), head(nullptr), tail(nullptr) {}

    void insertElement(int row, int col, T value)
    {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
        {
            // cerr << "Invalid row or column indices" << endl;
            return;
        }

        Node<T> *newNode = new Node<T>{row, col, value, nullptr};
        if (!head)
        {
            head = newNode;
            tail = newNode;
        }
        else
        {
            tail->next = newNode;
            tail = newNode;
        }
    }

    void add(SparseMatrix<T> &other)
    {
        if (rows != other.rows || cols != other.cols)
        {
            // cerr << "Matrix dimensions are not compatible for addition" << endl;
            return;
        }

        SparseMatrix<T> result(rows, cols);

        Node<T> *current1 = head;
        Node<T> *current2 = other.head;

        while (current1 != nullptr && current2 != nullptr)
        {
            if (current1->row < current2->row || (current1->row == current2->row && current1->col < current2->col))
            {
                result.insertElement(current1->row, current1->col, current1->value);
                current1 = current1->next;
            }
            else if (current2->row < current1->row || (current2->row == current1->row && current2->col < current1->col))
            {
                result.insertElement(current2->row, current2->col, current2->value);
                current2 = current2->next;
            }
            else
            {
                result.insertElement(current1->row, current1->col, current1->value + current2->value);
                current1 = current1->next;
                current2 = current2->next;
            }
        }

        while (current1 != nullptr)
        {
            result.insertElement(current1->row, current1->col, current1->value);
            current1 = current1->next;
        }

        while (current2 != nullptr)
        {
            result.insertElement(current2->row, current2->col, current2->value);
            current2 = current2->next;
        }

        result.display();
    }

    void multiply(SparseMatrix<T> &other)
    {
        if (cols != other.rows)
        {
            // cerr << "Matrix dimensions are not compatible for multiplication" << endl;
            return;
        }

        SparseMatrix<T> intermediate(rows, other.cols);
        SparseMatrix<T> result(rows, other.cols);

        Node<T> *current1 = head;
        Node<T> *current2 = other.head;

        while (current1 != nullptr)
        {
            while (current2 != nullptr)
            {
                if (current1->col == current2->row)
                {
                    intermediate.insertElement(current1->row, current2->col, current1->value * current2->value);
                }
                current2 = current2->next;
            }
            current1 = current1->next;
            current2 = other.head;
        }

        Node<T> *current3 = intermediate.head;
        Node<T> *prev = nullptr;
        Node<T> *temp = nullptr;

        while (current3 != nullptr)
        {
            temp = current3->next;
            prev = current3;
            while (temp != nullptr)
            {
                if (current3->row == temp->row && current3->col == temp->col)
                {
                    current3->value += temp->value;
                    prev->next = temp->next;
                    delete temp;
                    temp = prev->next;
                }
                else
                {
                    prev = temp;
                    temp = temp->next;
                }
            }
            current3 = current3->next;
        }

        result.head = indicesMergeSort(intermediate.head);
        result.display();
    }

    Node<T> *indicesMerge(Node<T> *left, Node<T> *right)
    {
        Node<T> *result = nullptr;

        if (!left)
            return right;
        if (!right)
            return left;

        if (left->row < right->row || (left->row == right->row && left->col < right->col))
        {
            result = left;
            result->next = indicesMerge(left->next, right);
        }
        else
        {
            result = right;
            result->next = indicesMerge(left, right->next);
        }

        return result;
    }

    Node<T> *indicesMergeSort(Node<T> *head)
    {
        if (!head || !head->next)
            return head;

        Node<T> *mid = head;
        Node<T> *fast = head->next;

        while (fast)
        {
            fast = fast->next;
            if (fast)
            {
                mid = mid->next;
                fast = fast->next;
            }
        }

        Node<T> *left = head;
        Node<T> *right = mid->next;
        mid->next = nullptr;

        left = indicesMergeSort(left);
        right = indicesMergeSort(right);

        return indicesMerge(left, right);
    }

    void transpose()
    {
        SparseMatrix<T> result(cols, rows);
        Node<T> *current = head;

        while (current)
        {
            result.insertElement(current->col, current->row, current->value);
            current = current->next;
        }

        result.head = indicesMergeSort(result.head);
        result.display();
    }

    void display()
    {
        Node<T> *current = head;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (current && current->row == i && current->col == j)
                {
                    cout << current->value << " ";
                    current = current->next;
                }
                else
                {
                    cout << "0 ";
                }
            }
            cout << '\n';
        }
    }

    // ~SparseMatrix() {
    //     while (head) {
    //         Node<T> *temp = head;
    //         head = head->next;
    //         delete temp;
    //     }
    // }
};

template <typename T>
class SparseMatrixArray
{
    T **trio;
    int rows, cols, size;
    const static int MAX = 8192; // capacity

public:
    SparseMatrixArray(int r, int c)
    {
        trio = new T *[MAX];
        for (int i = 0; i < MAX; i++)
        {
            trio[i] = new T[3];
        }
        rows = r;
        cols = c;
        size = 0;
    }

    void insertElmArr(int r, int c, T val)
    {
        if (r > rows || c > cols)
        {
            // cout << "Wrong entry";
            return;
        }
        else
        {
            trio[size][0] = r;
            trio[size][1] = c;
            trio[size][2] = val;
            size++;
        }
    }

    void display()
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                bool found = false;

                for (int k = 0; k < size; k++)
                {
                    if (trio[k][0] == i && trio[k][1] == j)
                    {
                        cout << trio[k][2] << " ";
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    cout << "0 ";
                }
            }
            cout << '\n';
        }
    }

    void add(SparseMatrixArray<T> b)
    {
        if (rows != b.rows || cols != b.cols)
        {
            // cout << "Matrices can't be added";
            return;
        }
        else
        {
            SparseMatrixArray<T> result(rows, cols);
            int idx = 0;
            int idxb = 0;

            while (idx < size && idxb < b.size)
            {
                if ((trio[idx][0] == b.trio[idxb][0] && trio[idx][1] < b.trio[idxb][1]) || trio[idx][0] < b.trio[idxb][0])
                {
                    result.insertElmArr(trio[idx][0], trio[idx][1], trio[idx][2]);
                    idx++;
                }
                else if ((trio[idx][0] == b.trio[idxb][0] && trio[idx][1] > b.trio[idxb][1]) || trio[idx][0] > b.trio[idxb][0])
                {
                    result.insertElmArr(b.trio[idxb][0], b.trio[idxb][1], b.trio[idxb][2]);
                    idxb++;
                }
                else
                {
                    T sum = trio[idx][2] + b.trio[idxb][2];
                    if (sum != 0)
                    {
                        result.insertElmArr(trio[idx][0], trio[idx][1], sum);
                    }
                    idx++;
                    idxb++;
                }
            }

            while (idx < size)
            {
                result.insertElmArr(trio[idx][0], trio[idx][1], trio[idx][2]);
                idx++;
            }

            while (idxb < b.size)
            {
                result.insertElmArr(b.trio[idxb][0], b.trio[idxb][1], b.trio[idxb][2]);
                idxb++;
            }

            result.display();
        }
    }

    SparseMatrixArray<T> transpose()
    {
        SparseMatrixArray<T> result(cols, rows);
        result.size = size;
        int *indices = new int[cols + 1];
        int *countarray = new int[cols + 1];
        for (int i = 1; i <= cols; i++)
        {
            indices[i] = 0;
            countarray[i] = 0;
        }

        for (int i = 0; i < size; i++)
        {
            countarray[trio[i][1]]++;
        }

        for (int i = 1; i <= cols; i++)
        {
            indices[i] = indices[i - 1] + countarray[i - 1];
        }
        for (int i = 0; i < size; i++)
        {
            int rowidx = indices[trio[i][1]]++;
            result.trio[rowidx][0] = trio[i][1];
            result.trio[rowidx][1] = trio[i][0];
            result.trio[rowidx][2] = trio[i][2];
        }
        return result;
    }

    void multiply(SparseMatrixArray<T> b)
    {
        SparseMatrixArray<T> result(rows, b.cols);

        if (cols != b.rows)
        {
            // cout << "Can't multiply, Invalid dimensions";
            return;
        }

        b = b.transpose();

        int idx = 0;
        int idxb = 0;

        while (idx < size)
        {
            int rw = trio[idx][0];
            while (idxb < b.size)
            {
                T sum = 0;
                int clm = b.trio[idxb][0];
                int iteridx = idx;
                int iteridxb = idxb;

                while (iteridx < size && iteridxb < b.size && trio[iteridx][0] == rw && b.trio[iteridxb][0] == clm)
                {
                    if (trio[iteridx][1] < b.trio[iteridxb][1])
                    {
                        iteridx++;
                    }
                    else if (trio[iteridx][1] > b.trio[iteridxb][1])
                    {
                        iteridxb++;
                    }
                    else
                    {
                        sum += trio[iteridx][2] * b.trio[iteridxb][2];
                        iteridx++;
                        iteridxb++;
                    }
                }

                if (sum != 0)
                {
                    result.insertElmArr(rw, clm, sum);
                }

                while (idxb < b.size && b.trio[idxb][0] == clm)
                {
                    idxb++;
                }
            }

            while (idx < size && trio[idx][0] == rw)
            {
                idx++;
                idxb = 0;
            }
        }

        result.display();
    }
};

int main()
{
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int T;
    cin >> T;
    while (T--)
    {
        int ds, op;
        cin >> ds;
        cin >> op;
        int N1, M1;
        cin >> N1 >> M1;
        if (ds == 2)
        {
            SparseMatrix<int> matrix1(N1, M1);

            for (int i = 0; i < N1; i++)
            {
                for (int j = 0; j < M1; j++)
                {
                    int value;
                    cin >> value;
                    if (value != 0)
                    {
                        matrix1.insertElement(i, j, value);
                    }
                }
            }

            if (op != 2)
            {
                int N2, M2;
                cin >> N2 >> M2;
                SparseMatrix<int> matrix2(N2, M2);

                for (int i = 0; i < N2; i++)
                {
                    for (int j = 0; j < M2; j++)
                    {
                        int value;
                        cin >> value;
                        if (value != 0)
                        {
                            matrix2.insertElement(i, j, value);
                        }
                    }
                }

                if (op == 1)
                {
                    matrix1.add(matrix2);
                }
                else
                {
                    matrix1.multiply(matrix2);
                }
            }
            else
            {
                matrix1.transpose();
            }
            // matrix1.display();
            // matrix2.display();
            // matrix1.add(matrix2);
            // matrix1.multiply(matrix2);
        }
        else
        {
            SparseMatrixArray<int> matrix1(N1, M1);

            for (int i = 0; i < N1; i++)
            {
                for (int j = 0; j < M1; j++)
                {
                    int value;
                    cin >> value;
                    if (value != 0)
                    {
                        matrix1.insertElmArr(i, j, value);
                    }
                }
            }

            if (op != 2)
            {
                int N2, M2;
                cin >> N2 >> M2;
                SparseMatrixArray<int> matrix2(N2, M2);

                for (int i = 0; i < N2; i++)
                {
                    for (int j = 0; j < M2; j++)
                    {
                        int value;
                        cin >> value;
                        if (value != 0)
                        {
                            matrix2.insertElmArr(i, j, value);
                        }
                    }
                }

                if (op == 1)
                {
                    matrix1.add(matrix2);
                }
                else
                {
                    matrix1.multiply(matrix2);
                }
            }
            else
            {
                SparseMatrixArray<int> trans = matrix1.transpose();
                trans.display();
            }
        }
    }
    return 0;
}