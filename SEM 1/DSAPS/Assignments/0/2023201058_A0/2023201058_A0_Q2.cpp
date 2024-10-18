#include <iostream>
using namespace std;

int main() 
{
    int N, S, J;
    cin >> N >> S >> J;
    int carr[N], jarr[J], idx = S;
    for(int i = 0; i < N; i++)
    {
        cin >> carr[i];
    }
    for(int j = 0; j < J; j++)
    {
        cin >> jarr[j];
    }

    cout << carr[S];
    for(int j = 0; j < J; j++)
    {
        idx = (idx + jarr[j]) % N;
        cout << " " << carr[idx];
    }    
    return 0;
}