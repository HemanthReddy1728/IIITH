#include <iostream>
using namespace std;

struct TreeNode
{
    long long max_length;
    long long ways;

    TreeNode() : max_length(0), ways(0) {}
};

TreeNode *segment_tree;

long long rankElements(long long *arr, long long n)
{
    long long *sorted_arr = new long long[n];
    for (long long i = 0; i < n; i++)
    {
        sorted_arr[i] = arr[i];
    }

    // Sort the temporary array
    for (long long i = 0; i < n; i++)
    {
        for (long long j = i + 1; j < n; j++)
        {
            if (sorted_arr[i] > sorted_arr[j])
            {
                swap(sorted_arr[i], sorted_arr[j]);
            }
        }
    }

    long long rank[10001];
    long long max_rank = 0;

    for (long long i = 0; i < 10001; i++)
    {
        rank[i] = -1;
    }

    for (long long i = 0; i < n; i++)
    {
        if (rank[sorted_arr[i] + 5000] == -1)
        {
            rank[sorted_arr[i] + 5000] = max_rank++;
        }
    }

    for (long long i = 0; i < n; i++)
    {
        arr[i] = rank[arr[i] + 5000];
    }

    delete[] sorted_arr;
    return max_rank;
}

TreeNode chooseBest(TreeNode &left, TreeNode &right)
{
    TreeNode result;

    long long max_length_left = left.max_length, ways_left = left.ways, max_length_right = right.max_length, ways_right = right.ways;

    if (max_length_left > max_length_right)
    {
        result.ways = ways_left;
        result.max_length = max_length_left;
    }
    else if (max_length_left < max_length_right)
    {
        result.ways = ways_right;
        result.max_length = max_length_right;
    }
    else
    { // same length, so we will add up the ways
        result.ways = ways_left + ways_right;
        result.max_length = max_length_left;
    }

    return result;
}

void updateSegmentTree(long long start, long long end, long long parent, long long element, long long max_length, long long ways)
{
    if (start == end)
    {
        if (segment_tree[parent].max_length == max_length)
        {
            segment_tree[parent].ways += ways;
        }
        else
        { 
            segment_tree[parent].ways = ways;
            segment_tree[parent].max_length = max_length;
        }
        return;
    }
    long long mid = (start + end) / 2;

    if (element <= mid)
    {
        updateSegmentTree(start, mid, 2 * parent + 1, element, max_length, ways);
    }
    else
    {
        updateSegmentTree(mid + 1, end, 2 * parent + 2, element, max_length, ways);
    }

    segment_tree[parent] = chooseBest(segment_tree[2 * parent + 1], segment_tree[2 * parent + 2]);
}

TreeNode queryMaxLen(long long start, long long end, long long query_start, long long query_end, long long parent)
{
    if (start > query_end || end < query_start)
    {
        return TreeNode();
    }
    if (start >= query_start && end <= query_end)
    {
        return segment_tree[parent];
    }
    long long mid = (start + end) / 2;
    TreeNode left = queryMaxLen(start, mid, query_start, query_end, 2 * parent + 1);
    TreeNode right = queryMaxLen(mid + 1, end, query_start, query_end, 2 * parent + 2);
    return chooseBest(left, right);
}

long long countNumberOfLIS(long long *arr, long long n)
{
    long long max_rank = rankElements(arr, n); 
    segment_tree = new TreeNode[4 * max_rank + 5];

    for (long long i = 0; i < n; i++)
    {
        long long max_length = 1; 
        long long ways = 1;       

        if (arr[i] >= 0)
        {
            TreeNode info = queryMaxLen(0, max_rank, 0, arr[i], 0);
            if (info.max_length + 1 > max_length)
            { 
                max_length = info.max_length + 1;
                ways = info.ways;
            }
        }

        updateSegmentTree(0, max_rank, 0, arr[i], max_length, ways); 
    }

    long long result = segment_tree[0].ways % 1000000007;
    delete[] segment_tree;
    return result;
}

int main()
{
    
    
    long long n;
    cin >> n; 

    long long *arr = new long long[n];
    for (long long i = 0; i < n; i++)
    {
        cin >> arr[i]; 
    }

    // Function Call
    cout << countNumberOfLIS(arr, n) << endl;

    delete[] arr; 
    
    return 0;
}
