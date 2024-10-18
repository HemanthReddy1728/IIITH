#include <iostream>
using namespace std;

int main() 
{
    int tc;
    cin >> tc;
    for(int i = 0; i < tc; i++)
    {
        string str1, str2;
        cin >> str1 >> str2;
        if (str1.size() != str2.size())
        {
            cout << "NO" << endl;
        }

        else
        {
            int arr[26] = {0}, chk[26] = {0}, lnt = str1.size(), flag = 0;

            for (int j = 0; j < lnt; j++)
            {
                int ascii1mod = ((int) str1[j]) % 97, ascii2mod = ((int) str2[j]) % 97;
                arr[ascii1mod] += 1;
                arr[ascii2mod] -= 1;
            }

            for (int k = 0; k < 26; k++)
            {
                if (arr[k] != chk[k])
                {
                    cout << "NO" << endl;
                    flag = 1;
                    break;
                }
            }
            
            if (flag == 0)
            {
                cout << "YES" << endl;
            }
        }
    }
    return 0;
}