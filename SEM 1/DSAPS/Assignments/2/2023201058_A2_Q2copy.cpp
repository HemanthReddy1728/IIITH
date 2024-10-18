#include <iostream>
using namespace std;

// Structure to represent a non-zero element in the matrix
template <typename T>
struct MatrixElement
{
    int row, col;
    T value;
    MatrixElement *next;
};

// Class to represent a sparse matrix using a linked list
template <typename T>
class SparseMatrix
{
private:
    int numRows, numCols;
    MatrixElement<T> *head;

public:
    SparseMatrix(int numRows, int numCols) : numRows(numRows), numCols(numCols), head(nullptr) {}

    // Function to add a non-zero element to the matrix
    void insertElement(int row, int col, T value)
    {
        if (row < 0 || row >= numRows || col < 0 || col >= numCols)
        {
            cerr << "Invalid row or column indices" << endl;
            return;
        }

        MatrixElement<T> *newElement = new MatrixElement<T>{row, col, value, nullptr};
        if (!head)
        {
            head = newElement;
        }
        else
        {
            MatrixElement<T> *current = head;
            while (current->next)
            {
                current = current->next;
            }
            current->next = newElement;
        }
    }

    // Function to perform matrix addition and return a new sparse matrix
    void add(SparseMatrix<T> &other)
    {
        if (numRows != other.numRows || numCols != other.numCols)
        {
            cerr << "Matrix dimensions are not compatible for addition" << endl;
        }

        SparseMatrix<T> result(numRows, numCols);

        MatrixElement<T> *current1 = head;
        MatrixElement<T> *current2 = other.head;

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

        result.display();
    }

    // Function to perform matrix multiplication and return a new sparse matrix
    void multiply(SparseMatrix<T> &other)
    {
        if (numCols != other.numRows)
        {
            cerr << "Matrix dimensions are not compatible for multiplication" << endl;
        }

        SparseMatrix<T> intermediate(numRows, other.numCols);
        SparseMatrix<T> result(numRows, other.numCols);

        T sum = 0;
        MatrixElement<T> *current1 = head;
        MatrixElement<T> *current2 = other.head;

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

        MatrixElement<T> *current3 = intermediate.head;
        MatrixElement<T> *prev = nullptr;
        MatrixElement<T> *temp = nullptr;

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

        current3 = intermediate.head;
        result.head = indicesMergeSort(current3);
        result.display();
    }

    // Function to merge two sorted linked lists based on row and column indices
    MatrixElement<T> *indicesMerge(MatrixElement<T> *left, MatrixElement<T> *right)
    {
        MatrixElement<T> *result = nullptr;

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

    // Function to perform merge sort on the linked list based on row and column indices
    MatrixElement<T> *indicesMergeSort(MatrixElement<T> *head)
    {
        if (!head || !head->next)
            return head;

        MatrixElement<T> *mid = head;
        MatrixElement<T> *fast = head->next;

        while (fast)
        {
            fast = fast->next;
            if (fast)
            {
                mid = mid->next;
                fast = fast->next;
            }
        }

        MatrixElement<T> *left = head;
        MatrixElement<T> *right = mid->next;
        mid->next = nullptr;

        left = indicesMergeSort(left);
        right = indicesMergeSort(right);

        return indicesMerge(left, right);
    }

    void transpose()
    {
        SparseMatrix<T> result(numCols, numRows);
        MatrixElement<T> *current = head;

        while (current)
        {
            result.insertElement(current->col, current->row, current->value);
            current = current->next;
        }

        result.head = indicesMergeSort(result.head); // Sort the transposed matrix

        result.display();
    }

    // Function to display the sparse matrix
    void display()
    {
        MatrixElement<T> *current = head;
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
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
            cout << endl;
        }
    }

    // Destructor to free memory
    ~SparseMatrix()
    {
        while (head)
        {
            MatrixElement<T> *temp = head;
            head = head->next;
            delete temp;
        }
    }
};

template <typename T>
class SparseMatrixArray
{
    T **elements;
    int numRows, numCols, size;
    const static int MAX = 100000; // capacity

public:
    SparseMatrixArray(int rows, int cols)
    {
        elements = new T *[MAX];
        for (int i = 0; i < MAX; i++)
        {
            elements[i] = new T[3];
        }
        numRows = rows;
        numCols = cols;
        size = 0;
    }

    void insertElementArray(int row, int col, T value)
    {
        if (row >= numRows || col >= numCols)
        {
            // cout << "Wrong entry";
            return;
        }
        else
        {
            elements[size][0] = row;
            elements[size][1] = col;
            elements[size][2] = value;
            size++;
        }
    }

    void display()
    {
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                bool found = false;

                for (int k = 0; k < size; k++)
                {
                    if (elements[k][0] == i && elements[k][1] == j)
                    {
                        cout << elements[k][2] << " ";
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    cout << "0 ";
                }
            }
            cout << endl;
        }
    }

    void add(SparseMatrixArray<T> b)
    {
        if (numRows != b.numRows || numCols != b.numCols)
        {
            // cout << "Matrices can't be added";
            return;
        }
        else
        {
            SparseMatrixArray<T> result(numRows, numCols);
            int idx = 0;
            int idxb = 0;

            while (idx < size && idxb < b.size)
            {
                if ((elements[idx][0] == b.elements[idxb][0] && elements[idx][1] < b.elements[idxb][1]) || elements[idx][0] < b.elements[idxb][0])
                {
                    result.insertElementArray(elements[idx][0], elements[idx][1], elements[idx][2]);
                    idx++;
                }
                else if ((elements[idx][0] == b.elements[idxb][0] && elements[idx][1] > b.elements[idxb][1]) || elements[idx][0] > b.elements[idxb][0])
                {
                    result.insertElementArray(b.elements[idxb][0], b.elements[idxb][1], b.elements[idxb][2]);
                    idxb++;
                }
                else
                {
                    T sum = elements[idx][2] + b.elements[idxb][2];
                    if (sum != 0)
                    {
                        result.insertElementArray(elements[idx][0], elements[idx][1], sum);
                    }
                    idx++;
                    idxb++;
                }
            }

            while (idx < size)
                result.insertElementArray(elements[idx][0], elements[idx][1], elements[idx++][2]);

            while (idxb < b.size)
                result.insertElementArray(b.elements[idxb][0], b.elements[idxb][1], b.elements[idxb++][2]);

            result.display();
        }
    }

    SparseMatrixArray<T> transpose()
    {
        SparseMatrixArray<T> result(numCols, numRows);
        result.size = size;
        int *indices = new int[numCols + 1];
        int *countArray = new int[numCols + 1];
        for (int i = 1; i <= numCols; i++)
        {
            indices[i] = 0;
            countArray[i] = 0;
        }

        for (int i = 0; i < size; i++)
        {
            countArray[elements[i][1]]++;
        }

        for (int i = 1; i <= numCols; i++)
        {
            indices[i] = indices[i - 1] + countArray[i - 1];
        }
        for (int i = 0; i < size; i++)
        {
            int rowIdx = indices[elements[i][1]]++;
            result.elements[rowIdx][0] = elements[i][1];
            result.elements[rowIdx][1] = elements[i][0];
            result.elements[rowIdx][2] = elements[i][2];
        }
        return result;
    }

    void multiply(SparseMatrixArray<T> b)
    {
        SparseMatrixArray<T> result(numRows, b.numCols);
        // row-row multiplication
        if (numCols != b.numRows)
        {
            // cout << "Can't multiply, Invalid dimensions";
            return;
        }

        b = b.transpose();
        int idx, idxb;

        for (idx = 0; idx < size;)
        {
            int rw = elements[idx][0];
            for (idxb = 0; idxb < b.size;)
            {
                T sum = 0;
                int clm = b.elements[idxb][0];
                int iteridx = idx;
                int iteridxb = idxb;

                while (iteridx < size && iteridxb < b.size && elements[iteridx][0] == rw && b.elements[iteridxb][0] == clm)
                {
                    if (elements[iteridx][1] < b.elements[iteridxb][1])
                    {
                        iteridx++;
                    }
                    else if (elements[iteridx][1] > b.elements[iteridxb][1])
                    {
                        iteridxb++;
                    }
                    else
                    {
                        sum += elements[iteridx][2] * b.elements[iteridxb][2];
                    }
                }

                if (sum != 0)
                {
                    result.insertElementArray(rw, clm, sum);
                }

                while (idxb < b.size && b.elements[idxb][0] == clm)
                {
                    idxb++;
                }
            }

            while (idx < size && elements[idx][0] == rw)
            {
                idx++;
            }
        }
        result.display();
    }
};

int main()
{
    int dataType, operation;
    cin >> dataType;
    cin >> operation;
    int numRows1, numCols1;
    cin >> numRows1 >> numCols1;

    if (dataType == 2)
    {
        SparseMatrix<int> matrix1(numRows1, numCols1);

        for (int i = 0; i < numRows1; i++)
        {
            for (int j = 0; j < numCols1; j++)
            {
                int value;
                cin >> value;
                if (value != 0)
                {
                    matrix1.insertElement(i, j, value);
                }
            }
        }

        if (operation != 2)
        {
            int numRows2, numCols2;
            cin >> numRows2 >> numCols2;
            SparseMatrix<int> matrix2(numRows2, numCols2);

            for (int i = 0; i < numRows2; i++)
            {
                for (int j = 0; j < numCols2; j++)
                {
                    int value;
                    cin >> value;
                    if (value != 0)
                    {
                        matrix2.insertElement(i, j, value);
                    }
                }
            }

            if (operation == 1)
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
    }
    else
    {
        SparseMatrixArray<int> matrix1(numRows1, numCols1);

        for (int i = 0; i < numRows1; i++)
        {
            for (int j = 0; j < numCols1; j++)
            {
                int value;
                cin >> value;
                if (value != 0)
                {
                    matrix1.insertElementArray(i, j, value);
                }
            }
        }

        if (operation != 2)
        {
            int numRows2, numCols2;
            cin >> numRows2 >> numCols2;
            SparseMatrixArray<int> matrix2(numRows2, numCols2);

            for (int i = 0; i < numRows2; i++)
            {
                for (int j = 0; j < numCols2; j++)
                {
                    int value;
                    cin >> value;
                    if (value != 0)
                    {
                        matrix2.insertElementArray(i, j, value);
                    }
                }
            }

            if (operation == 1)
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
            matrix1.transpose().display();
        }
    }

    return 0;
}
