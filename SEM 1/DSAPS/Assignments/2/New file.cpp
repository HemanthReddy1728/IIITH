#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdio.h>
#define max 1
using namespace std;

bool comparesort(const int *a, const int *b)
{

    if (a[0] < b[0])
        return true;
    else if (a[0] == b[0])
    {

        return a[1] < b[1];
    }
    else
    {
        return false;
    }
}

class sparsearr
{

public:
    int **val;
    int row;
    int col;
    int count;
    int size;

    void reserve(int newcap)
    {

        int **datanew = new int *[newcap];

        for (int i = 0; i < newcap; i++)
        {

            datanew[i] = new int[3];
        }

        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < 3; j++)
            {

                datanew[i][j] = val[i][j];
            }
        }
        delete[] val;
        val = datanew;
        size = newcap;
    }

    sparsearr(int r, int c)
    {
        val = new int *[max];
        row = r;
        col = c;
        count = 0;
        size = max;

        for (int i = 0; i < max; i++)
        {

            val[i] = new int[3];
        }
    }

    void insertnodearr(int r, int c, int elem)
    {

        if (count >= size)
            reserve(2 * size);

        val[count][0] = r;

        val[count][1] = c;

        val[count][2] = elem;

        count++;
    }
    void displaylocal()
    {

        int k = 0;

        // Traverse the list.
        while (k < count)
        {
            cout << val[k][0] << " " << val[k][1] << " " << val[k][2] << " \n";

            k++;
        }
    }

    void sortthis() { ::sort(val, val + count, comparesort); }

    void display(int n1, int m1)
    {

        int tt = 0, t1 = 0, t2 = 0;
        while (t1 < n1)
        {
            t2 = 0;
            while (t2 < m1)

            {
                // if (tt!= (n1*m1) && tt < (n1*m1) && t2 < m1 && t1 < n1)

                //{
                if (tt < count && val[tt][0] == t1 && val[tt][1] == t2)
                {
                    cout << val[tt][2] << " ";
                    tt++;
                    t2++;
                }

                else
                {
                    cout << "0" << " ";

                    t2++;
                }
                //}
            }
            cout << "\n";

            t1++;
        }
    }

    void displaytrans(int n1, int m1)
    {

        int tt = 0, t1 = 0, t2 = 0;
        while (t1 < m1)
        {
            t2 = 0;
            while (t2 < n1)

            {
                while (tt < count && t2 < n1 && t1 < m1)

                {
                    if (val[tt][0] == t1 && val[tt][1] == t2)
                    {
                        cout << val[tt][2] << " ";
                        tt++;
                        t2++;
                    }

                    else
                    {
                        cout << "0" << " ";
                        t2++;
                    }
                }

                cout << "\n";
            }

            t1++;
        }
    }
};

class Node
{

public:
    int r;
    int c;
    int data;
    Node *next;

    Node()
    {

        data = 0;
        next = NULL;
    }

    Node(int data, int row, int col)
    {

        this->data = data;
        this->r = row;
        this->c = col;
        this->next = NULL;
    }
};

class ll
{

public:
    Node *head;
    Node *end;

    ll()
    {
        head = NULL;
        end = NULL;
    }

    void insertnode(int data, int row, int col)
    {

        Node *latest = new Node(data, row, col);

        if (head == NULL)
        {

            head = latest;
            end = latest;
            return;
        }

        else
        {

            Node *temp = end;

            temp->next = latest;
            temp = temp->next;
            end = temp;
        }
    }

    void display()
    {

        Node *temp = head;

        // Check for empty list.
        if (head == NULL)
        {
            cout << "List empty" << endl;
            return;
        }

        // Traverse the list.
        while (temp != NULL)
        {
            cout << temp->data << " " << temp->r << " " << temp->c << " " << "\n";
            temp = temp->next;
        }
    }

    void displayadd(int n1, int m1)
    {

        Node *t = head;

        for (int i = 0; i < n1; i++)

        {
            for (int j = 0; j < m1; j++)
            {

                if (t != NULL && t->r == i && t->c == j)
                {
                    cout << t->data << " ";
                    t = t->next;
                }

                else
                {
                    cout << "0" << " ";
                }
            }
            cout << "\n";
        }
    }

    ll mergesort(ll &res3)
    {

        Node *t = res3.head;

        if (t == NULL || t->next == NULL)
        {

            return res3;
        }

        ll lefthalf;
        ll righthalf;
        ll result;

        divlist(res3, lefthalf, righthalf);

        lefthalf = mergesort(lefthalf);
        righthalf = mergesort(righthalf);

        result = merge(lefthalf, righthalf);

        return result;
    }
    void divlist(ll &result, ll &lefthalf, ll &righthalf)
    {

        Node *sl = result.head;

        Node *fa = result.head->next;

        while (fa != NULL && fa->next != NULL)
        {

            sl = sl->next;
            fa = fa->next->next;
        }

        lefthalf.head = result.head;
        righthalf.head = sl->next;
        sl->next = NULL;
    }

    ll merge(ll &lefthalf, ll &righthalf)
    {

        ll finresult;
        Node *t1 = lefthalf.head;
        Node *t2 = righthalf.head;

        while (t1 != NULL && t2 != NULL)
        {

            if (t1->r < t2->r || (t1->r == t2->r) && t1->c < t2->c)
            {

                finresult.insertnode(t1->data, t1->r, t1->c);
                t1 = t1->next;
            }

            else
            {

                finresult.insertnode(t2->data, t2->r, t2->c);
                t2 = t2->next;
            }
        }

        while (t1 != NULL)
        {
            finresult.insertnode(t1->data, t1->r, t1->c);
            t1 = t1->next;
        }

        while (t2 != NULL)
        {
            finresult.insertnode(t2->data, t2->r, t2->c);
            t2 = t2->next;
        }

        return finresult;
    }

    void dotrans(int n1, int m1)
    {

        Node *t = head;
        ll translist;

        while (t != NULL)
        {
            translist.insertnode(t->data, t->c, t->r);
            t = t->next;
        }
        ll trans2;
        trans2 = translist.mergesort(translist);
        trans2.displayadd(m1, n1);
    }
};

int main()
{

    int t;
    int op;
    int n1, m1, n2, m2;

    cin >> t;
    cin >> op;
    cin >> n1;
    cin >> m1;

    if (t == 2)
    {
        ll list;
        ll list2;

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < m1; j++)
            {

                int d;
                cin >> d;

                if (d != 0)
                {

                    list.insertnode(d, i, j);
                }
            }
        }

        if (op == 2)
        {

            list.dotrans(n1, m1);
        }

        if (op == 1 || op == 3)
        {
            cin >> n2;
            cin >> m2;

            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < m2; j++)
                {

                    int d;
                    cin >> d;

                    if (d != 0)
                    {

                        list2.insertnode(d, i, j);
                    }
                }
            }
        }

        if (op == 1)
        {

            Node *t1 = list.head;
            Node *t2 = list2.head;
            ll res;

            while (t1 != NULL && t2 != NULL)
            {

                if (t1->r == t2->r && t1->c == t2->c)
                {

                    res.insertnode(t1->data + t2->data, t1->r, t1->c);
                    t1 = t1->next;
                    t2 = t2->next;
                }
                else if (t1->r < t2->r || (t1->r == t2->r && t1->c < t2->c))
                {
                    res.insertnode(t1->data, t1->r, t1->c);
                    t1 = t1->next;
                }

                else
                {

                    res.insertnode(t2->data, t2->r, t2->c);
                    t2 = t2->next;
                }
            }

            while (t1 != NULL)
            {
                res.insertnode(t1->data, t1->r, t1->c);
                t1 = t1->next;
            }

            while (t2 != NULL)
            {
                res.insertnode(t2->data, t2->r, t2->c);
                t2 = t2->next;
            }

            cout << "\n";
            res.displayadd(n1, m1);
        }

        if (op == 3)
        {

            Node *t1 = list.head;
            Node *t2 = list2.head;
            ll res3;

            while (t1 != NULL)
            {
                while (t2 != NULL)
                {
                    if (t1->c == t2->r)
                    {

                        res3.insertnode(t1->data * t2->data, t1->r, t2->c);
                    }
                    t2 = t2->next;
                }
                t2 = list2.head;
                t1 = t1->next;
            }

            Node *t3 = res3.head;
            Node *curres = res3.head;
            Node *prev = NULL;

            while (curres != NULL)
            {
                Node *temp = curres->next;
                prev = curres;

                while (temp != NULL)
                {

                    if (curres->r == temp->r && curres->c == temp->c)

                    {

                        curres->data += temp->data;
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

                curres = curres->next;
            }

            // res3.display();
            ll fin;
            fin = res3.mergesort(res3);
            fin.displayadd(n1, m2);
            // fin.display();
        }
    }

    else if (t == 1)
    {

        sparsearr arr(n1, m1);
        sparsearr arr2(n2, m2);

        for (int i = 0; i < n1; i++)
        {

            for (int j = 0; j < m1; j++)

            {
                int va;
                cin >> va;

                if (va != 0)
                    arr.insertnodearr(i, j, va);
            }
        }

        // arr.display(n1,m1);

        if (op == 1)
        {

            cin >> n2;
            cin >> m2;

            for (int i = 0; i < n2; i++)
            {

                for (int j = 0; j < m2; j++)

                {
                    int va;
                    cin >> va;

                    if (va != 0)
                        arr2.insertnodearr(i, j, va);
                }
            }

            sparsearr ressum(n1, m1);

            int l = 0, m = 0;

            while (l < arr.count && m < arr2.count)
            {
                if (arr.val[l][0] == arr2.val[m][0] && arr.val[l][1] == arr2.val[m][1])
                {
                    ressum.insertnodearr(arr.val[l][0], arr.val[l][1], arr.val[l][2] + arr2.val[m][2]);
                    l++;
                    m++;
                }
                else if (arr.val[l][0] < arr2.val[m][0] || (arr.val[l][0] == arr2.val[m][0] && (arr.val[l][1] < arr2.val[m][1])))
                {
                    ressum.insertnodearr(arr.val[l][0], arr.val[l][1], arr.val[l][2]);
                    l++;
                }

                else
                {
                    ressum.insertnodearr(arr2.val[m][0], arr2.val[m][1], arr2.val[m][2]);
                    m++;
                }
            }

            while (l < arr.count)
            {
                ressum.insertnodearr(arr.val[l][0], arr.val[l][1], arr.val[l][2]);
                l++;
            }

            while (m < arr2.count)
            {
                ressum.insertnodearr(arr2.val[m][0], arr2.val[m][1], arr2.val[m][2]);
                m++;
            }

            cout << "\n";
            ressum.display(n1, m1);
        }

        if (op == 2)
        {

            sparsearr datatrans(n1, m1);
            for (int i = 0; i < arr.count; i++)
            {
                datatrans.insertnodearr(arr.val[i][1], arr.val[i][0], arr.val[i][2]);
            }

            datatrans.sortthis();

            datatrans.display(m1, n1);
        }

        if (op == 3)
        {

            cin >> n2;
            cin >> m2;

            //  Node *t2 = list2.head;
            // ll res3;
            for (int i = 0; i < n2; i++)
            {

                for (int j = 0; j < m2; j++)

                {
                    int va;
                    cin >> va;

                    if (va != 0)
                        arr2.insertnodearr(i, j, va);
                }
            }
            //  arr.displaylocal();

            int mat1 = 0, mat2 = 0;

            sparsearr matmult(n1, m2);

            while (mat1 < arr.count)
            {
                while (mat2 < arr2.count)
                {
                    if (arr.val[mat1][1] == arr2.val[mat2][0])
                    {

                        matmult.insertnodearr(arr.val[mat1][0], arr2.val[mat2][1], arr.val[mat1][2] * arr2.val[mat2][2]);
                    }
                    mat2++;
                }
                mat2 = 0;
                mat1++;
            }

            matmult.sortthis();
            // matmult.displaylocal();

            int sum = 0;
            int matf = 0;
            sparsearr matfin(n1, m2);

            while (matf < matmult.count)
            {

                int rr = matmult.val[matf][0];
                int cc = matmult.val[matf][1];
                int el = matmult.val[matf][2];
                sum = matmult.val[matf][2];

                while (matf + 1 < matmult.count && matmult.val[matf + 1][0] == rr && matmult.val[matf + 1][1] == cc)
                {

                    matf++;
                    sum += matmult.val[matf][2];
                }
                matfin.insertnodearr(rr, cc, sum);
                matf++;
                sum = 0;
            }

            matfin.display(n1, m2);
        }
    }
}

// list.display();
// list2.display();
