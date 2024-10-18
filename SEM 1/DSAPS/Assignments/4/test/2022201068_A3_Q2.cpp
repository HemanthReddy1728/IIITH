#include <time.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Size of the smaller file to be created out of a bigger file
#define CHUNK_SIZE 1000000
// #define CHUNK_SIZE 10000000
// #define CHUNK_SIZE 100000000

using namespace std;

/*
!Notes:
    -> the primary goal of an external sorting algorithm is to minimize the number of times information must be read from or written to disk

    ->Under good conditions, reading from a file in sequential order is more efficient than reading blocks in random order.


    Divide Files:
        1. open input file
        2. Read first k bytes
        3. While not end of the file:
            open new output file
            write k bytes
            close output file
            read next k block

*/

/// @brief Structure of the Pair for Min Heap
struct Pair
{
    long long value;
    long long ind;
};

/// @brief MinHeap Class
class MinHeap
{
private:
    vector<Pair> heap;

    void swap(Pair *a, Pair *b)
    {
        Pair c = (*a);
        *a = (*b);
        *b = (c);
    }

public:
    /// @brief Method to insert an element in the Min Heap
    /// @param key
    /// @param _ind
    void push(long long key, long long _ind)
    {

        heap.push_back({key, _ind});

        long long ind = heap.size() - 1;

        while (ind != 0)
        {
            if (heap[ind].value < heap[(ind - 1) / 2].value)
            {
                swap(&heap[ind], &heap[(ind - 1) / 2]);
                ind = (ind - 1) / 2;
            }
            else
            {
                break;
            }
        }
    }

    /// @brief Method to Pop an Element from the Heap
    /// @return Minimum Value Pair
    Pair pop()
    {
        if (heap.empty())
            return {INT64_MIN, INT64_MIN};

        if (heap.size() == 1)
        {
            Pair ans = {heap[0].value, heap[0].ind};
            heap.pop_back();
            return ans;
        }

        long long minvalue = heap[0].value;
        long long minind = heap[0].ind;
        swap(&heap[0], &heap[heap.size() - 1]);
        heap.pop_back();

        long long ind = 0;

        while ((2 * ind + 1) < heap.size() and (2 * ind + 2) < heap.size())
        {
            long long leftchild = 2 * ind + 1;
            long long rightchild = 2 * ind + 2;

            long long smallchild;

            if (heap[leftchild].value < heap[ind].value and heap[leftchild].value <= heap[rightchild].value)
            {
                smallchild = leftchild;
            }
            else if (heap[rightchild].value < heap[ind].value and heap[rightchild].value <= heap[leftchild].value)
            {
                smallchild = rightchild;
            }
            else
            {
                break;
            }

            swap(&heap[ind], &heap[smallchild]);

            ind = smallchild;
        }

        if (heap.size() == 2)
        {
            if (heap[ind].value > heap[ind + 1].value)
                swap(&heap[ind], &heap[ind + 1]);
        }

        return {minvalue, minind};
    }

    /// @brief Method to check if the Min Heap is empty or not
    /// @return Boolean True if Min Heap is empty otherwise false
    bool empty() { return heap.empty(); }
};

/// @brief Divide Step of Divide and Conquer Algorithm to Divide a larger file
/// into smaller files of CHUNK_SIZE
/// @param inputfilename
/// @return Number of Files Created
long long dividefiles(string inputfilename)
{
    long long ind = 0;

    ifstream inputfile(inputfilename, std::ios_base::in);

    for (ind = 0; inputfile.peek() != EOF; ind++)
    {

        vector<long long> buff;

        long long a;

        for (long long i = 0; i < CHUNK_SIZE; i++)
        {
            if (inputfile >> a)
            {
                buff.push_back(a);
            }
            else
            {
                break;
            }
        }

        sort(buff.begin(), buff.end());
        string chunkfilename = "temp" + to_string(ind) + ".txt";
        ofstream chunkfile{chunkfilename};

        for (auto x : buff)
        {
            chunkfile << x << '\n';
        }

        chunkfile.close();
    }
    inputfile.close();

    return ind;
}

/// @brief Combine Step of Divide and Conquer Algorithm to combine smaller
/// sorted file into a final output file
/// @param totalfiles
/// @param outfilename
void mergefiles(long long totalfiles, string outfilename)
{
    vector<ifstream> filesarr;
    vector<bool> finish(totalfiles, false);

    for (long long i = 0; i < totalfiles; i++)
    {
        string chunkfilename = "temp" + to_string(i) + ".txt";
        filesarr.push_back(ifstream(chunkfilename, std::ios_base::in));
    }

    MinHeap minh;

    for (long long i = 0; i < totalfiles; i++)
    {
        long long curr;
        if (!finish[i] and filesarr[i] >> curr)
        {
            minh.push(curr, i);
        }
        else
        {
            filesarr[i].close();
            finish[i] = true;
        }
    }

    ofstream outfile{outfilename};

    while (!minh.empty())
    {
        Pair top = minh.pop();
        outfile << top.value << "\n";
        long long i = top.ind;
        long long curr;
        if (!finish[i] and filesarr[i] >> curr)
        {
            minh.push(curr, i);
        }
        else
        {
            filesarr[i].close();
            finish[i] = true;
        }
    }
    outfile.close();
}

/// @brief Utility to Clean the smaller created files
/// @param totalfiles
void cleanfiles(long long totalfiles)
{
    for (long long i = 0; i < totalfiles; i++)
    {
        string chunkfilename = "temp" + to_string(i) + ".txt";
        remove(chunkfilename.c_str());
    }
}

/// @brief Driver Function
/// @param argc
/// @param argv
/// @return Status of the Program
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Please enter two arguments only" << endl;
        return 1;
    }

    auto start = clock();

    string inputfilename(argv[1]);
    string outputfilename(argv[2]);
    long long totalfiles = dividefiles(inputfilename);

    cout << "\033[1;36m" << "Number of integers in a temporary file: " << "\033[m" << CHUNK_SIZE << endl;

    cout << "\033[1;31m" << "Number of temporary files created: " << "\033[m" << totalfiles << endl;
    mergefiles(totalfiles, outputfilename);

    cleanfiles(totalfiles);

    auto end = clock();

    double totaltime = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "\033[1;32m" << "Total time: " << "\033[m" << fixed << setprecision(2) << totaltime << "s" << endl;

    return 0;
}
