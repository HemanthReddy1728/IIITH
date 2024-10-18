#include <iostream>
using namespace std;

template <typename T>
class sparse_matrix
{
    // Maximum number of elements in matrix
    const static int MAX = 100000;

    // Double-pointer initialized by
    // the constructor to store
    // the triple-represented form
    T **data;

    // dimensions of matrix
    int row, col;

    // total number of elements in matrix
    int len;

public:
    sparse_matrix(int r, int c)
    {
        // initialize row
        row = r;

        // initialize col
        col = c;

        // initialize length to 0
        len = 0;

        // Array of Pointer to make a matrix
        data = new T *[MAX];

        // Array representation
        // of sparse matrix
        //[,0] represents row
        //[,1] represents col
        //[,2] represents value
        for (int i = 0; i < MAX; i++)
            data[i] = new T[3];
    }

    // insert elements into sparse matrix
    void insert(int r, int c, T val)
    {
        // invalid entry
        if (r > row || c > col)
        {
            cout << "Wrong entry";
        }
        else
        {
            // insert row value
            data[len][0] = r;

            // insert col value
            data[len][1] = c;

            // insert element's value
            data[len][2] = val;

            // increment number of data in matrix
            len++;
        }
    }

    void add(sparse_matrix<T> b)
    {
        // if matrices don't have the same dimensions
        if (row != b.row || col != b.col)
        {
            cout << "Matrices can't be added";
        }
        else
        {
            int apos = 0, bpos = 0;
            sparse_matrix<T> result(row, col);

            while (apos < len && bpos < b.len)
            {
                // if b's row and col are smaller
                if (data[apos][0] > b.data[bpos][0] || (data[apos][0] == b.data[bpos][0] && data[apos][1] > b.data[bpos][1]))
                {
                    // insert the smaller value into the result
                    result.insert(b.data[bpos][0], b.data[bpos][1], b.data[bpos][2]);
                    bpos++;
                }
                // if a's row and col are smaller
                else if (data[apos][0] < b.data[bpos][0] || (data[apos][0] == b.data[bpos][0] && data[apos][1] < b.data[bpos][1]))
                {
                    // insert the smaller value into the result
                    result.insert(data[apos][0], data[apos][1], data[apos][2]);
                    apos++;
                }
                else
                {
                    // add the values as row and col are the same
                    T addedval = data[apos][2] + b.data[bpos][2];
                    if (addedval != 0)
                        result.insert(data[apos][0], data[apos][1], addedval);
                    apos++;
                    bpos++;
                }
            }

            // insert remaining elements
            while (apos < len)
                result.insert(data[apos][0], data[apos][1], data[apos++][2]);

            while (bpos < b.len)
                result.insert(b.data[bpos][0], b.data[bpos][1], b.data[bpos++][2]);

            // print the result
            result.print();
        }
    }

    sparse_matrix<T> transpose()
    {
        // new matrix with inversed row X col
        sparse_matrix<T> result(col, row);

        // same number of elements
        result.len = len;

        // to count the number of elements in each column
        int *count = new int[col + 1];

        // initialize all to 0
        for (int i = 1; i <= col; i++)
            count[i] = 0;

        for (int i = 0; i < len; i++)
            count[data[i][1]]++;

        int *index = new int[col + 1];

        // to count the number of elements having
        // col smaller than a particular i

        // as there is no col with value < 0
        index[0] = 0;

        // initialize the rest of the indices
        for (int i = 1; i <= col; i++)
            index[i] = index[i - 1] + count[i - 1];

        for (int i = 0; i < len; i++)
        {
            // insert a data at rpos and increment its value
            int rpos = index[data[i][1]]++;

            // transpose row=col
            result.data[rpos][0] = data[i][1];

            // transpose col=row
            result.data[rpos][1] = data[i][0];

            // same value
            result.data[rpos][2] = data[i][2];
        }

        // the above method ensures
        // sorting of the transpose matrix
        // according to row-col value
        return result;
    }

    void multiply(sparse_matrix<T> b)
    {
        if (col != b.row)
        {
            // Invalid multiplication
            cout << "Can't multiply, Invalid dimensions";
            return;
        }

        // transpose b to compare row
        // and col values and to add them at the end
        b = b.transpose();
        int apos, bpos;

        // result matrix of dimension row X b.col
        // however b has been transposed,
        // hence row X b.row
        sparse_matrix<T> result(row, b.row);

        // iterate over all elements of A
        for (apos = 0; apos < len;)
        {
            // current row of the result matrix
            int r = data[apos][0];

            // iterate over all elements of B
            for (bpos = 0; bpos < b.len;)
            {
                // current column of the result matrix
                // data[,0] used as b is transposed
                int c = b.data[bpos][0];

                // temporary pointers created to add all
                // multiplied values to obtain the current
                // element of the result matrix
                int tempa = apos;
                int tempb = bpos;

                T sum = 0;

                // iterate over all elements with
                // the same row and col value
                // to calculate result[r]
                while (tempa < len && data[tempa][0] == r && tempb < b.len && b.data[tempb][0] == c)
                {
                    if (data[tempa][1] < b.data[tempb][1])
                        // skip a
                        tempa++;
                    else if (data[tempa][1] > b.data[tempb][1])
                        // skip b
                        tempb++;
                    else
                        // same col, so multiply and increment
                        sum += data[tempa++][2] * b.data[tempb++][2];
                }

                // insert sum obtained in result[r]
                // if it's not equal to 0
                if (sum != 0)
                    result.insert(r, c, sum);

                while (bpos < b.len && b.data[bpos][0] == c)
                    // jump to the next column
                    bpos++;
            }
            while (apos < len && data[apos][0] == r)
                // jump to the next row
                apos++;
        }
        result.print();
    }

    // printing matrix
    void print()
    {
        // cout << "\nDimension: " << row << "x" << col;
        // cout << "\nSparse Matrix: \nRow\tColumn\tValue\n";

        for (int i = 0; i < len; i++)
        {
            cout << data[i][0] << "\t " << data[i][1] << "\t " << data[i][2] << endl;
        }
    }
};

// Driver Code
int main()
{
    // // create two sparse matrices and insert values
    // sparse_matrix<int> a(4, 4);
    // sparse_matrix<int> b(4, 4);

    // a.insert(1, 2, 10);
    // a.insert(1, 4, 12);
    // a.insert(3, 3, 5);
    // a.insert(4, 1, 15);
    // a.insert(4, 2, 12);
    // b.insert(1, 3, 8);
    // b.insert(2, 4, 23);
    // b.insert(3, 3, 9);
    // b.insert(4, 1, 20);
    // b.insert(4, 2, 25);

    // // Output result
    // cout << "Addition: ";
    // a.add(b);
    // cout << "\nMultiplication: ";
    // a.multiply(b);
    // cout << "\nTranspose: ";
    // sparse_matrix<int> atranspose = a.transpose();
    // atranspose.print();



    return 0;
}
