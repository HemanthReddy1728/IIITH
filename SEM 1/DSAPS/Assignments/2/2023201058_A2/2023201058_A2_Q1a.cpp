#include <iostream>
using namespace std;

class PriorityQueue
{
public:
    int *heap;
    int capacity;
    int size;

    void printDetails()
    {
        cout << "Capacity: " << capacity << " ";
        cout << "Size: " << size << " ";
        cout << "Heap: ";
        for (int i = 0; i < size; i++)
        {
            cout << heap[i] << " ";
        }
        cout << endl;
    }

    void minHeapifyUp(int index)
    {
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            if (heap[index] < heap[parentIndex])
            {
                swap(heap[index], heap[parentIndex]);
                index = parentIndex;
            }
            else
            {
                break;
            }
        }
        // printDetails();
    }

    void minHeapifyDown(int index)
    {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int smallest = index;

        if (leftChild < size && heap[leftChild] < heap[smallest])
        {
            smallest = leftChild;
        }

        if (rightChild < size && heap[rightChild] < heap[smallest])
        {
            smallest = rightChild;
        }

        if (smallest != index)
        {
            swap(heap[index], heap[smallest]);
            minHeapifyDown(smallest);
        }
        // printDetails();
    }

    void maxHeapifyUp(int index)
    {
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            if (heap[index] > heap[parentIndex])
            {
                swap(heap[index], heap[parentIndex]);
                index = parentIndex;
            }
            else
            {
                break;
            }
        }
        // printDetails();
    }

    void maxHeapifyDown(int index)
    {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int largest = index;

        if (leftChild < size && heap[leftChild] > heap[largest])
        {
            largest = leftChild;
        }

        if (rightChild < size && heap[rightChild] > heap[largest])
        {
            largest = rightChild;
        }

        if (largest != index)
        {
            swap(heap[index], heap[largest]);
            maxHeapifyDown(largest);
        }
        // printDetails();
    }

    PriorityQueue() : capacity(1), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    PriorityQueue(int capacity) : capacity(capacity), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    ~PriorityQueue()
    {
        delete[] heap;
    }

    int getSize()
    {
        printDetails();
        return size;
    }

    void push(int el)
    {
        if (size == capacity)
        {
            // Double the capacity if the current capacity is reached.
            int newCapacity = capacity * 2;
            int *newHeap = new int[newCapacity];

            // Copy elements from the old heap to the new heap.
            for (int i = 0; i < capacity; i++)
            {
                newHeap[i] = heap[i];
            }

            // Delete the old heap and update capacity.
            delete[] heap;
            heap = newHeap;
            capacity = newCapacity;
            printDetails();
        }

        heap[size] = el;
        // minHeapifyUp(size);
        maxHeapifyUp(size);
        size++;
        printDetails();
    }

    int top()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return -1; // Return a default value or handle the error as needed
        }
        printDetails();
        return heap[0];
    }

    void pop()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return;
        }
        swap(heap[0], heap[size - 1]);
        size--;
        // minHeapifyDown(0);
        maxHeapifyDown(0);
        printDetails();
    }

    bool empty()
    {
        printDetails();
        return size == 0;
    }
};

class MaxPriorityQueue
{
public:
    int *heap;
    int capacity;
    int size;

    void printDetails()
    {
        cout << "Capacity: " << capacity << " ";
        cout << "Size: " << size << " ";
        cout << "Heap: ";
        for (int i = 0; i < size; i++)
        {
            cout << heap[i] << " ";
        }
        cout << endl;
    }

    void maxHeapifyUp(int index)
    {
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            if (heap[index] > heap[parentIndex])
            {
                swap(heap[index], heap[parentIndex]);
                index = parentIndex;
            }
            else
            {
                break;
            }
        }
        // printDetails();
    }

    void maxHeapifyDown(int index)
    {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int largest = index;

        if (leftChild < size && heap[leftChild] > heap[largest])
        {
            largest = leftChild;
        }

        if (rightChild < size && heap[rightChild] > heap[largest])
        {
            largest = rightChild;
        }

        if (largest != index)
        {
            swap(heap[index], heap[largest]);
            maxHeapifyDown(largest);
        }
        // printDetails();
    }

    MaxPriorityQueue() : capacity(1), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    MaxPriorityQueue(int capacity) : capacity(capacity), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    ~MaxPriorityQueue()
    {
        delete[] heap;
    }

    int getSize()
    {
        printDetails();
        return size;
    }

    void push(int el)
    {
        if (size == capacity)
        {
            // Double the capacity if the current capacity is reached.
            int newCapacity = capacity * 2;
            int *newHeap = new int[newCapacity];

            // Copy elements from the old heap to the new heap.
            for (int i = 0; i < capacity; i++)
            {
                newHeap[i] = heap[i];
            }

            // Delete the old heap and update capacity.
            delete[] heap;
            heap = newHeap;
            capacity = newCapacity;
            
        }

        heap[size] = el;
        maxHeapifyUp(size);
        size++;
        
        printDetails();
    }

    int top()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return -1; // Return a default value or handle the error as needed
        }
        printDetails();
        return heap[0];
    }

    void pop()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return;
        }
        swap(heap[0], heap[size - 1]);
        size--;
        maxHeapifyDown(0);
        printDetails();
    }

    bool empty()
    {
        printDetails();
        return size == 0;
    }
};

class MinPriorityQueue
{
public:
    int *heap;
    int capacity;
    int size;

    void printDetails()
    {
        cout << "Capacity: " << capacity << " ";
        cout << "Size: " << size << " ";
        cout << "Heap: ";
        for (int i = 0; i < size; i++)
        {
            cout << heap[i] << " ";
        }
        cout << endl;
    }

    void minHeapifyUp(int index)
    {
        while (index > 0)
        {
            int parentIndex = (index - 1) / 2;
            if (heap[index] < heap[parentIndex])
            {
                swap(heap[index], heap[parentIndex]);
                index = parentIndex;
            }
            else
            {
                break;
            }
        }
        // printDetails();
    }

    void minHeapifyDown(int index)
    {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int smallest = index;

        if (leftChild < size && heap[leftChild] < heap[smallest])
        {
            smallest = leftChild;
        }

        if (rightChild < size && heap[rightChild] < heap[smallest])
        {
            smallest = rightChild;
        }

        if (smallest != index)
        {
            swap(heap[index], heap[smallest]);
            minHeapifyDown(smallest);
        }
        // printDetails();
    }

    MinPriorityQueue() : capacity(1), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    MinPriorityQueue(int capacity) : capacity(capacity), size(0)
    {
        heap = new int[capacity];
        printDetails();
    }

    ~MinPriorityQueue()
    {
        delete[] heap;
    }

    int getSize()
    {
        printDetails();
        return size;
    }

    void push(int el)
    {
        if (size == capacity)
        {
            // Double the capacity if the current capacity is reached.
            int newCapacity = capacity * 2;
            int *newHeap = new int[newCapacity];

            // Copy elements from the old heap to the new heap.
            for (int i = 0; i < capacity; i++)
            {
                newHeap[i] = heap[i];
            }

            // Delete the old heap and update capacity.
            delete[] heap;
            heap = newHeap;
            capacity = newCapacity;
            printDetails();
        }

        heap[size] = el;
        minHeapifyUp(size);
        size++;
        printDetails();
    }

    int top()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return -1; // Return a default value or handle the error as needed
        }
        printDetails();
        return heap[0];
    }

    void pop()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            printDetails();
            return;
        }
        swap(heap[0], heap[size - 1]);
        size--;
        minHeapifyDown(0);
        printDetails();
    }

    bool empty()
    {
        printDetails();
        return size == 0;
    }
};

// int main()
// {
//     PriorityQueue pq;
//     // MaxPriorityQueue pq;
//     // MinPriorityQueue pq;

//     pq.push(3);
//     pq.push(1);
//     pq.push(10);
//     pq.push(4);
//     pq.push(2);
//     pq.push(8);

//     cout << "Size: " << pq.getSize() << endl;
//     cout << "Top: " << pq.top() << endl;

//     pq.pop();
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;

//     pq.pop();
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;

//     pq.pop();
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;
//     pq.pop();
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;
//     pq.pop();
//     pq.push(8);
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;
//     pq.pop();
//     cout << "Size after pop: " << pq.getSize() << endl;
//     cout << "Top after pop: " << pq.top() << endl;
//     return 0;

// }
