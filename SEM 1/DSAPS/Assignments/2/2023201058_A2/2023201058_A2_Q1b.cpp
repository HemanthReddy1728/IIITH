#include <iostream>
using namespace std;

template <typename T>
class MyDeque
{
private:
    T *Arr;
    long long Cpct;
    long long Frontindex;
    long long Rearindex;
    long long MyDequesize;

public:
    void printdetails()
    {
        return; // Comment this return statement to use printdetails function
        cout << Cpct << " " << Frontindex << " " << Rearindex << " " << MyDequesize << endl;
        if (Frontindex == -1 && Rearindex == -1)
        {
            return;
        }
        long long f = Frontindex;
        if (MyDequesize)
        {
            for (long long i = 0; i < MyDequesize; i++, f = (f + 1) % Cpct)
            {
                cout << Arr[f] << " ";
            }
        }
        cout << endl;
        return;
    }

    MyDeque() : Cpct(0), Frontindex(-1), Rearindex(-1), MyDequesize(0)
    {
        Arr = new T[Cpct];
        printdetails();
    }

    MyDeque(long long n) : Cpct(n), Frontindex(0), Rearindex(n - 1), MyDequesize(n)
    {
        Arr = new T[Cpct];
        for (long long i = 0; i < n; i++)
        {
            Arr[i] = T();
        }
        printdetails();
    }

    MyDeque(long long n, T x) : Cpct(n), Frontindex(0), Rearindex(n - 1), MyDequesize(n)
    {
        Arr = new T[Cpct];
        for (long long i = 0; i < n; i++)
        {
            Arr[i] = x;
        }
        printdetails();
    }

    // Destructor creates seg faults
    // ~MyDeque()
    // {
    //     delete[] Arr;
    // }

    bool push_back(T x)
    {
        if (Frontindex == Rearindex && Rearindex == -1)
        {
            T *arr1 = new T[1];
            arr1[0] = x;
            // delete [] Arr;
            Arr = arr1;
            Frontindex = 0;
            Rearindex = 0;
            MyDequesize++;
            Cpct = 1;
            printdetails();
            return true;
        }

        if (full())
        {
            if (MyDequesize == Cpct)
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
        }
        if (empty())
        {
            Frontindex = Rearindex = 0;
        }
        else
        {
            Rearindex = (Rearindex + 1) % Cpct;
        }
        Arr[Rearindex] = x;
        MyDequesize++;
        printdetails();
        return true;
    }

    bool pop_back()
    {
        if (empty())
        {
            return false;
        }
        if (Frontindex == Rearindex)
        {
            Frontindex = Rearindex = -1;
        }
        else
        {
            Rearindex = (Rearindex - 1 + Cpct) % Cpct;
        }
        MyDequesize--;
        printdetails();
        return true;
    }

    bool push_front(T x)
    {
        if (Frontindex == Rearindex && Rearindex == -1)
        {
            T *arr1 = new T[1];
            arr1[0] = x;
            // delete [] Arr;
            Arr = arr1;
            Frontindex = 0;
            Rearindex = 0;
            MyDequesize++;
            Cpct = 1;
            printdetails();
            return true;
        }
        if (full())
        {
            if (MyDequesize == Cpct)
            {
                long long newCapacity;
                if (Cpct == 0)
                {
                    newCapacity = 1;
                }
                else
                {
                    newCapacity = Cpct * 2;
                }
                reserve(newCapacity);
            }
        }
        if (empty())
        {
            Frontindex = Rearindex = 0;
        }
        else
        {
            Frontindex = (Frontindex - 1 + Cpct) % Cpct;
        }
        Arr[Frontindex] = x;
        MyDequesize++;
        printdetails();
        return true;
    }

    bool pop_front()
    {
        if (empty())
        {
            return false;
        }
        if (Frontindex == Rearindex)
        {
            Frontindex = Rearindex = -1;
        }
        else
        {
            Frontindex = (Frontindex + 1) % Cpct;
        }
        MyDequesize--;
        printdetails();
        return true;
    }

    T front()
    {
        if (empty())
        {
            return T();
        }
        return Arr[Frontindex];
    }

    T back()
    {
        if (empty())
        {
            return T();
        }
        return Arr[Rearindex];
    }

    T operator[](long long n)
    {
        if (n >= 0)
        {
            if (n >= MyDequesize)
            {
                printdetails();
                return T();
            }
            printdetails();
            return Arr[(Frontindex + n) % Cpct];
        }
        else
        {
            n = -n;
            if (n > MyDequesize)
            {
                printdetails();
                return T();
            }
            printdetails();
            return Arr[(Rearindex - n + 1 + Cpct) % Cpct];
        }
    }

    bool empty()
    {
        printdetails();
        return MyDequesize == 0;
        // return Frontindex == -1;
    }

    bool full()
    {
        printdetails();
        return MyDequesize == Cpct;
    }

    long long size()
    {
        printdetails();
        return MyDequesize;
    }

    void clear()
    {
        Frontindex = Rearindex = -1;
        MyDequesize = 0;
        printdetails();
    }

    long long capacity()
    {
        printdetails();
        return Cpct;
    }

    void resize(long long n)
    {
        T *newArr;
        if (n == MyDequesize)
        {
            printdetails();
            return;
        }
        if (n > Cpct)
        {
            newArr = new T[n];
            Cpct = n;
        }
        else
        {
            newArr = new T[Cpct];
        }
        Frontindex = 0;
        for (long long i = 0; i < n; i++)
        {
            if (i < MyDequesize)
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            else
            {
                newArr[i] = T();
            }
        }
        MyDequesize = n;
        Frontindex = 0;
        Rearindex = MyDequesize - 1;
        // delete[] Arr;
        Arr = newArr;
        printdetails();
    }

    void resize(long long n, T d)
    {
        T *newArr;
        if (n == Rearindex)
        {
            printdetails();
            return;
        }
        if (n > Cpct)
        {
            newArr = new T[n];
            Cpct = n;
        }
        else
        {
            newArr = new T[Cpct];
        }
        Frontindex = 0;
        for (long long i = 0; i < n; i++)
        {
            if (i < MyDequesize)
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            else
            {
                newArr[i] = d;
            }
        }
        MyDequesize = n;
        Frontindex = 0;
        Rearindex = MyDequesize - 1;
        // delete[] Arr;
        Arr = newArr;
        printdetails();
    }

    void reserve(long long n)
    {
        if (n > Cpct)
        {
            T *newArr = new T[n];
            for (long long i = 0; i < MyDequesize; i++)
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            Cpct = n;
            Frontindex = 0;
            Rearindex = MyDequesize - 1;
            // delete[] Arr;
            Arr = newArr;
        }
        printdetails();
    }

    void shrink_to_fit()
    {
        if (Cpct > MyDequesize)
        {
            T *newArr = new T[MyDequesize];
            for (long long i = 0; i < MyDequesize; i++)
            {
                newArr[i] = Arr[(Frontindex + i) % Cpct];
            }
            Cpct = MyDequesize;
            Frontindex = 0;
            Rearindex = MyDequesize - 1;
            // delete[] Arr;
            Arr = newArr;
        }
        printdetails();
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
        // printDetails();
    }

    MaxPriorityQueue(int capacity) : capacity(capacity), size(0)
    {
        heap = new int[capacity];
        // printDetails();
    }

    ~MaxPriorityQueue()
    {
        delete[] heap;
    }

    int getSize()
    {
        // printDetails();
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
            heap = nullptr;
            heap = newHeap;
            capacity = newCapacity;
            // printDetails();
        }

        heap[size] = el;
        maxHeapifyUp(size);
        size++;
        // printDetails();
    }

    int top()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            // printDetails();
            return -1; // Return a default value or handle the error as needed
        }
        // printDetails();
        return heap[0];
    }

    void pop()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            // printDetails();
            return;
        }
        swap(heap[0], heap[size - 1]);
        size--;
        maxHeapifyDown(0);
        // printDetails();
    }

    bool empty()
    {
        // printDetails();
        return size == 0;
    }

    // void remove(int el)
    // {
    //     // Find the index of the element to be removed
    //     int index = -1;
    //     for (int i = 0; i < size; i++)
    //     {
    //         if (heap[i] == el)
    //         {
    //             index = i;
    //             break;
    //         }
    //     }

    //     if (index == -1)
    //     {
    //         // printDetails();
    //         return; // Element not found in the heap
    //     }

    //     // Replace the element to be removed with the last element
    //     heap[index] = heap[size - 1];
    //     size--;

    //     // Fix the heap property
    //     maxHeapifyDown(index);
    //     // printDetails();
    // }
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
        // printDetails();
    }

    MinPriorityQueue(int capacity) : capacity(capacity), size(0)
    {
        heap = new int[capacity];
        // printDetails();
    }

    ~MinPriorityQueue()
    {
        delete[] heap;
    }

    int getSize()
    {
        // printDetails();
        return size;
    }

    void push(int el)
    {
        if (size == capacity)
        {

            // printDetails();
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
            // printDetails();
        }

        // printDetails();
        heap[size] = el;
        minHeapifyUp(size);
        size++;
        // printDetails();
    }

    int top()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            // printDetails();
            return -1; // Return a default value or handle the error as needed
        }
        // printDetails();
        return heap[0];
    }

    void pop()
    {
        if (size == 0)
        {
            // cout << "Priority queue is empty" << endl;
            // printDetails();
            return;
        }
        swap(heap[0], heap[size - 1]);
        size--;
        minHeapifyDown(0);
        // printDetails();
    }

    bool empty()
    {
        // printDetails();
        return size == 0;
    }

    // void remove(int el)
    // {
    //     // Find the index of the element to be removed
    //     int index = -1;
    //     for (int i = 0; i < size; i++)
    //     {
    //         if (heap[i] == el)
    //         {
    //             index = i;
    //             break;
    //         }
    //     }

    //     if (index == -1)
    //     {
    //         // printDetails();
    //         return; // Element not found in the heap
    //     }

    //     // Replace the element to be removed with the last element
    //     heap[index] = heap[size - 1];
    //     size--;

    //     // Fix the heap property
    //     minHeapifyDown(index);
    //     // printDetails();
    // }
};

MyDeque<double> medianStream(MyDeque<int> &inputNumbers)
{
    MyDeque<double> medianValues;
    MaxPriorityQueue greaterHalf, smallerHalf;
    int numberOfElements = inputNumbers.size();

    for (int i = 0; i < numberOfElements; i++)
    {
        smallerHalf.push(inputNumbers[i]);
        int tempstore = smallerHalf.top();
        smallerHalf.pop();

        greaterHalf.push(-1.0 * tempstore);
        if (greaterHalf.size > smallerHalf.size)
        {
            tempstore = greaterHalf.top();
            greaterHalf.pop();
            smallerHalf.push(-1.0 * tempstore);
        }
        if (greaterHalf.size != smallerHalf.size)
        {
            medianValues.push_back((double)smallerHalf.top() * 1.0);
        }
        else
        {
            medianValues.push_back((double)((smallerHalf.top() * 1.0 - greaterHalf.top() * 1.0) / 2.0));
        }
    }

    return medianValues;
}

MyDeque<double> medianSlidingWindow(MyDeque<int> &inputNumbers, int windowSize)
{
    MyDeque<double> windowMedians;
    int numberOfElements = inputNumbers.size();
    int elementFrequency[100001] = {0};                 
    MinPriorityQueue minHeap;                           
    MaxPriorityQueue maxHeap, greaterHalf, smallerHalf; 

    for (int i = 0; i < windowSize; i++)
    {
        maxHeap.push(inputNumbers[i]);
    }

    for (int i = 0; i < (windowSize / 2); i++)
    {
        minHeap.push(maxHeap.top());
        maxHeap.pop();
    }

    // & 1 -> even or odd check
    for (int i = windowSize; i < numberOfElements; i++)
    {
        if (windowSize & 1)
        { 
            windowMedians.push_back(maxHeap.top() * 1.0);
        }
        else
        {
            windowMedians.push_back(((double)maxHeap.top() + (double)minHeap.top()) / 2);
        }

        int previousElement = inputNumbers[i - windowSize], currentElement = inputNumbers[i], balance = 0; 

        if (previousElement <= maxHeap.top())
        { 
            balance--;
            if (previousElement == maxHeap.top())
            {
                maxHeap.pop();
            }
            else
            {
                elementFrequency[previousElement]++;
            }
        }
        else
        { 
            balance++;
            if (previousElement == minHeap.top())
                {
                    minHeap.pop();
                }
            else
                {
                    elementFrequency[previousElement]++;
                }
        }
        
        if (!maxHeap.empty() && currentElement <= maxHeap.top())
        { 
            balance++;
            maxHeap.push(currentElement);
        }
        else
        { 
            balance--;
            minHeap.push(currentElement);
        }

        if (balance < 0)
        {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
        else if (balance > 0)
        {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        }

        while (!maxHeap.empty() && elementFrequency[maxHeap.top()])
        {
            elementFrequency[maxHeap.top()]--;
            maxHeap.pop();
        }
        while (!minHeap.empty() && elementFrequency[minHeap.top()])
        {
            elementFrequency[minHeap.top()]--;
            minHeap.pop();
        }
    }

    if (windowSize & 1)
    {
        windowMedians.push_back(maxHeap.top() * 1.0);
    }
    else
    {
        windowMedians.push_back(((double)maxHeap.top() + (double)minHeap.top()) / 2.0);
    }

    return windowMedians;
}

int main()
{
    int n, k, cnt=0;
    cin >> n >> k;
    MyDeque<int> nums;
    for (int i = 0; i < n; i++)
    {
        int in;
        cin >> in;
        nums.push_back(in);
    }
    // for (int i = 0; i < n; i++)
    // {
    //     cout << nums[i] << " ";
    // }
    // cout << endl;

    MyDeque<double> medians = medianStream(nums);
    MyDeque<double> window_medians = medianSlidingWindow(nums, k);
    // for (int i = 0; i < medians.size(); i++)
    // {
    //     cout << medians[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < window_medians.size(); i++)
    // {
    //     cout << window_medians[i] << " ";
    // }
    // cout << endl;

    for(int i = k; i <= n; i++)
    {
        // cout << window_medians[i - k] << " " << medians[i - 1] << " " << nums[i] << " " << endl;
        if(window_medians[i-k] + medians[i-1] <= nums[i])
        {
            cnt++;
        }
    }
    cout << cnt << endl;
    return 0;
}
