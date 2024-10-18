#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ctime>
using namespace std;

#define CHUNK_SIZE 10000

struct Pair
{
    long long value;
    long long index;
};

class MinPriorityQueue
{
public:
    vector<Pair> heap;

    void swap(Pair *a, Pair *b)
    {
        Pair temp = *a;
        *a = *b;
        *b = temp;
    }

    bool isEmpty()
    {
        return heap.empty();
    }

    void insert(long long key, long long index)
    {
        heap.push_back({key, index});

        long long pos = heap.size() - 1;

        while (pos != 0)
        {
            if (heap[pos].value < heap[(pos - 1) / 2].value)
            {
                swap(&heap[pos], &heap[(pos - 1) / 2]);
                pos = (pos - 1) / 2;
            }
            else
            {
                break;
            }
        }
    }

    Pair removeMin()
    {
        if (heap.empty())
            return {INT64_MIN, INT64_MIN};

        if (heap.size() == 1)
        {
            Pair min = {heap[0].value, heap[0].index};
            heap.pop_back();
            return min;
        }

        long long minValue = heap[0].value;
        long long minIndex = heap[0].index;
        swap(&heap[0], &heap[heap.size() - 1]);
        heap.pop_back();

        long long pos = 0;

        while ((2 * pos + 1) < static_cast<long long>(heap.size()) and (2 * pos + 2) < static_cast<long long>(heap.size()))
        {
            long long smallerChild;
            long long leftChild = 2 * pos + 1;
            long long rightChild = 2 * pos + 2;


            if (heap[leftChild].value < heap[pos].value and heap[leftChild].value <= heap[rightChild].value)
            {
                smallerChild = leftChild;
            }
            else if (heap[rightChild].value < heap[pos].value and heap[rightChild].value <= heap[leftChild].value)
            {
                smallerChild = rightChild;
            }
            else
            {
                break;
            }

            swap(&heap[pos], &heap[smallerChild]);

            pos = smallerChild;
        }

        if (heap.size() == 2)
        {
            if (heap[pos].value > heap[pos + 1].value)
                swap(&heap[pos], &heap[pos + 1]);
        }

        return {minValue, minIndex};
    }
};

long long divideAndSortChunks(string inputFileName)
{
    long long chunkIndex = 0;

    ifstream inputFile(inputFileName, ios_base::in);

    for (chunkIndex = 0; inputFile.peek() != EOF; chunkIndex++)
    {
        vector<long long> chunk;
        long long value;

        for (long long i = 0; i < CHUNK_SIZE; i++)
        {
            if (inputFile >> value)
            {
                chunk.push_back(value);
            }
            else
            {
                break;
            }
        }

        sort(chunk.begin(), chunk.end());

        string chunkFileName = "temp" + to_string(chunkIndex) + ".txt";
        ofstream chunkFile{chunkFileName};

        for (auto x : chunk)
        {
            chunkFile << x << '\n';
        }

        chunkFile.close();
    }
    inputFile.close();

    return chunkIndex;
}

void mergeChunks(long long totalChunks, string outputFileName)
{
    vector<ifstream> chunkStreams;
    vector<bool> chunkFinished(totalChunks, false);

    for (long long i = 0; i < totalChunks; i++)
    {
        string chunkFileName = "temp" + to_string(i) + ".txt";
        chunkStreams.push_back(ifstream(chunkFileName, ios_base::in));
    }

    MinPriorityQueue minHeap;

    for (long long i = 0; i < totalChunks; i++)
    {
        long long current;
        if (!chunkFinished[i] and chunkStreams[i] >> current)
        {
            minHeap.insert(current, i);
        }
        else
        {
            chunkStreams[i].close();
            chunkFinished[i] = true;
        }
    }

    ofstream outputFile{outputFileName};

    while (!minHeap.isEmpty())
    {
        Pair top = minHeap.removeMin();
        outputFile << top.value << "\n";
        long long i = top.index;
        long long current;
        if (!chunkFinished[i] and chunkStreams[i] >> current)
        {
            minHeap.insert(current, i);
        }
        else
        {
            chunkStreams[i].close();
            chunkFinished[i] = true;
        }
    }
    outputFile.close();
}

void cleanUpChunks(long long totalChunks)
{
    for (long long i = 0; i < totalChunks; i++)
    {
        string chunkFileName = "temp" + to_string(i) + ".txt";
        remove(chunkFileName.c_str());
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Please provide input and output file names as arguments." << endl;
        return 1;
    }

    auto startTime = clock();

    string inputFileName(argv[1]);
    string outputFileName(argv[2]);
    long long totalChunks = divideAndSortChunks(inputFileName);

    cout << "Number of integers per temporary file : " << CHUNK_SIZE << endl;
    cout << "Number of temporary files created : " << totalChunks << endl;

    mergeChunks(totalChunks, outputFileName);

    cleanUpChunks(totalChunks);

    auto endTime = clock();

    double totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    cout << "Total time taken : " << fixed << setprecision(2) << totalTime << " seconds" << endl;

    return 0;
}
