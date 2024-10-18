#include <bits/stdc++.h>

using namespace std;

struct Pair {
    long long value;
    ifstream fs;
    long long ind;
};

class MinHeap {
   private:
    vector<Pair> arr;
    // long long size;
    // long long max_size;

    void swap(Pair *a, Pair *b) {
        Pair c = move(*a);
        *a = move(*b);
        *b = move(c);
    }

   public:
    // MinHeap(long long max_ele) {
    //     // max_size = max_ele;
    //     // size = 0;
    // }

    void insert(long long key, ifstream fs, long long _ind) {
        // if (size == max_size) return;

        arr.push_back({key, move(fs), _ind});

        // arr[size].value = key;
        // arr[size].fs = move(fs);
        // arr[size].ind = _ind;
        long long ind = arr.size() - 1;
        // size++;

        while (ind != 0) {
            if (arr[ind].value < arr[(ind - 1) / 2].value) {
                swap(&arr[ind], &arr[(ind - 1) / 2]);
                ind = (ind - 1) / 2;
            } else {
                break;
            }
        }
    }

    Pair remove() {
        if (arr.empty()) return {INT64_MIN};

        if (arr.size() == 1) {
            // size = 0;

            Pair ans = {arr[0].value, move(arr[0].fs), arr[0].ind};
            arr.pop_back();
            return ans;
        }

        long long maxval = arr[0].value;
        ifstream maxfs = move(arr[0].fs);
        long long maxind = arr[0].ind;
        // replace root by last element
        swap(&arr[0], &arr[arr.size() - 1]);
        arr.pop_back();
        // size--;

        long long ind = 0;

        while ((2 * ind + 1) < arr.size() and (2 * ind + 2) < arr.size()) {
            long long leftchild = 2 * ind + 1;
            long long rightchild = 2 * ind + 2;

            long long smallchild;

            if (arr[leftchild].value < arr[ind].value and
                arr[leftchild].value <= arr[rightchild].value) {
                smallchild = leftchild;
            } else if (arr[rightchild].value < arr[ind].value and
                       arr[rightchild].value < arr[leftchild].value) {
                smallchild = rightchild;
            } else {
                break;
            }

            swap(&arr[ind], &arr[smallchild]);

            ind = smallchild;
        }

        if (arr.size() == 2) {
            if (arr[ind].value < arr[ind + 1].value)
                swap(&arr[ind], &arr[ind + 1]);
        }

        return {maxval, move(maxfs), maxind};
    }

    bool empty() { return arr.empty(); }
};

int main() {
    MinHeap mini;
    ifstream fs("input.txt");
    mini.insert(123, move(fs), 0);
    mini.insert(138, move(fs), 2);
    mini.insert(923, move(fs), 4);
    mini.insert(18, move(fs), 6);
    mini.insert(91, move(fs), 7);
    mini.insert(1243, move(fs), 5);
    mini.insert(2108, move(fs), 2);
    mini.insert(3802, move(fs), 5);
    mini.insert(293, move(fs), 6);
    mini.insert(23, move(fs), 7);

    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;
    cout << mini.remove().value << endl;

    return 0;
}