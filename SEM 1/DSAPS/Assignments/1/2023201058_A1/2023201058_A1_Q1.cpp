#include <iostream>
#include <algorithm>

using namespace std;

string addnumstr(string numstr1, string numstr2)
{
    int l1 = numstr1.length() - 1, l2 = numstr2.length() - 1, carry = 0;
    if (l1 < l2) 
    {
        return addnumstr(numstr2, numstr1);
    }

    string sumstr;
    while (l1 >= 0 || l2 >= 0 || carry > 0)
    {
        int digit1, digit2, sum, unitplace;
        if ( l1 >= 0 )
        {
            digit1 = numstr1[l1] - '0';
        } 
        else
        {
            digit1 = 0;
        }
        l1--;

        if ( l2 >= 0 )
        {
            digit2 = numstr2[l2] - '0';
        } 
        else
        {
            digit2 = 0;
        }
        l2--;

        sum = digit1 + digit2 + carry;
        unitplace = sum % 10;
        carry = sum / 10;
        sumstr.push_back(char(unitplace + '0'));
    }

    reverse(sumstr.begin(), sumstr.end());
    return sumstr;
}

bool numcompareGE(string numstr1, string numstr2) 
{
    int l1 = numstr1.length(), l2 = numstr2.length();
    if (l1 > l2) 
    {
        return true;
    }
    else if (l1 < l2) 
    {
        return false;
    }
    else
    {
        for (int i = 0; i < l1; i++) 
        {
            if (numstr1[i] > numstr2[i]) 
            {
                return true;
            } 
            if (numstr1[i] < numstr2[i]) 
            {
                return false;
            }
        }
        return true; // Equal case
    }
}

string subnumstr(string numstr1, string numstr2) 
{
    string diffstr;
    if (!numcompareGE(numstr1, numstr2)) 
    {
        return "-" + subnumstr(numstr2, numstr1);
    }

    int l1 = numstr1.length() - 1, l2 = numstr2.length() - 1, borrow = 0;
    while (l1 >= 0 || l2 >= 0) 
    {
        int digit1, digit2, diff;

        if ( l1 >= 0 )
        {
            digit1 = numstr1[l1] - '0';
        } 
        else
        {
            digit1 = 0;
        }
        l1--;
        if ( l2 >= 0 )
        {
            digit2 = numstr2[l2] - '0';
        } 
        else
        {
            digit2 = 0;
        }
        l2--;

        diff = digit1 - digit2 - borrow;
        if (diff < 0) 
        {
            diff += 10;
            borrow = 1;
        } 
        else 
        {
            borrow = 0;
        }

        diffstr.push_back(char(diff + '0'));
    }
    while (diffstr.length() > 1 && diffstr.back() == '0') 
    {
        diffstr.pop_back();
    }

    reverse(diffstr.begin(), diffstr.end());
    return diffstr;
}

string mulnumstr(string numstr1, string numstr2) 
{
    string prodstr = "0";
    int l1 = numstr1.length() - 1, l2 = numstr2.length() - 1;

    for (int i = l2; i >= 0; i--) 
    {
        string subprodstr;
        int carry = 0;

        for (int j = l1; j >= 0; j--) 
        {
            int unitprod = (numstr1[j] - '0') * (numstr2[i] - '0') + carry;
            carry = unitprod / 10;
            subprodstr.push_back(char(unitprod % 10 + '0'));
        }
        if (carry > 0) 
        {
            subprodstr.push_back(char(carry + '0'));
        }
        reverse(subprodstr.begin(), subprodstr.end());
        for (int k = 0; k < l2 - i; k++) 
        {
            subprodstr.push_back('0');
        }
        prodstr = addnumstr(prodstr, subprodstr);
    }
    return prodstr;
}

string divnumstr(string numstr1, string numstr2) 
{
    string quotstr, rmdrstr;
    for (char digit : numstr1) 
    {
        rmdrstr.push_back(digit);
        int quotDigit = 0;
        
        while (numcompareGE(rmdrstr, numstr2)) 
        {
            rmdrstr = subnumstr(rmdrstr, numstr2);
            quotDigit++;
        }

        if (!quotstr.empty() || quotDigit > 0) 
        {
            quotstr.push_back(char(quotDigit + '0'));
        }
    }
    if (quotstr.empty()) 
    {
        quotstr = "0";
    }
    return quotstr;
}

string modnumstr(string numstr1, string numstr2) 
{
    string rmdrstr;
    for (char digit : numstr1) 
    {
        rmdrstr.push_back(digit);
        
        while (numcompareGE(rmdrstr, numstr2)) 
        {
            rmdrstr = subnumstr(rmdrstr, numstr2);
        }
    }
    return rmdrstr;
}

string expostr(string numstr1, string numstr2)
{
    if (numstr2 == "0")
    {
        return "1";
    }

    else if (numstr2 == "1")
    {
        return numstr1;
    }
    else if (modnumstr(numstr2, "2") == "0")
    {
        return mulnumstr(expostr(numstr1, divnumstr(numstr2, "2")), expostr(numstr1, divnumstr(numstr2, "2")));
    }
    else
    {
        return mulnumstr(numstr1, mulnumstr(expostr(numstr1, divnumstr(numstr2, "2")), expostr(numstr1, divnumstr(numstr2, "2"))));
    }
}

string gcdstr(string numstr1, string numstr2)
{
    if (numstr1 == numstr2)
    {
        return numstr1;
    }
    else if (numstr1 == "0" )
    {
        return numstr2;
    }
    else if (numstr2 == "0")
    {
        return numstr1;
    }
    else{
        while (numstr1 != numstr2)
        {   
            if (numstr1 == "0" )
            {
                return numstr2;
            }
            else if (numstr2 == "0")
            {
                return numstr1;
            }
            else if (numcompareGE(numstr1, numstr2))
            {
                numstr1 = modnumstr(numstr1, numstr2);
            }
            else
            {
                numstr2 = modnumstr(numstr2, numstr1);
            }
        }
        return numstr1;
    }
}

string factostr(string numstr1)
{
    if (numstr1 == "0" or numstr1 == "1")
    {
        return "1";
    }
    else
    {
        string front = "1", rearfact = "1";
        while(numcompareGE(numstr1, front))
        {
            rearfact = mulnumstr(front, rearfact);
            front = addnumstr(front, "1");
        }
        return rearfact;
    }
}

int main() 
{   
    // while (true)
    // {
        // cout << "Menu:" << endl;
        // cout << "0. Exit" << endl;
        // cout << "1. Addition(+), subtraction(-), multiplication(x, lowercase “X”), division(/)" << endl;
        // cout << "2. Exponentiation" << endl;
        // cout << "3. GCD of two numbers" << endl;
        // cout << "4. Factorial" << endl;

        // // d.printdetails();
        // cout << "Enter your choice: ";
        int choice;
        cin >> choice;
        // system("clear");

        string infixexpr = "", base = "", exponent = "", num1 = "", num2 = "", num3 = "";

        if (choice == 0) 
        {
            cout << "Exiting the program." << endl;
            return 0;
        }

        switch (choice) 
        {
            case 1:
            {
                cin >> infixexpr;
                long long operatorcount = 0, infixarraysize = 0;
                long long infixexprsize = infixexpr.size();
                for (long long i = 0; i < infixexprsize; i++)
                {
                    if (infixexpr[i] == '+' || infixexpr[i] == '-' || infixexpr[i] == 'x' || infixexpr[i] == '/')
                    {
                        operatorcount++;
                    }
                }
                   
                // infixexpr string to infixarray array

                infixarraysize = 1 + 2 * operatorcount;
                string infixarray[infixarraysize], stack[infixarraysize], postfixarray[infixarraysize];
                long long infixarrayptridx = 0, top = -1, postfixarrayptridx = -1;
                
                for (long long i = 0; i < infixexprsize; i++) 
                {
                    if (infixexpr[i] == '+' || infixexpr[i] == '-' || infixexpr[i] == 'x' || infixexpr[i] == '/')
                    {
                        infixarray[infixarrayptridx+1] = infixexpr[i];
                        infixarrayptridx+=2;
                    }
                    else
                    {
                        infixarray[infixarrayptridx] += infixexpr[i];
                    }
                }

                // infixarray array to postfixarray array

                for (int i = 0; i < infixarraysize; i++)
                {
                    if (infixarray[i] != "+" && infixarray[i] != "-" && infixarray[i] != "x" && infixarray[i] != "/")
                    {
                        postfixarray[postfixarrayptridx+1] = infixarray[i];
                        postfixarrayptridx++;
                    }
                    else if (top == -1 && (infixarray[i] == "+" || infixarray[i] == "-" || infixarray[i] == "x" || infixarray[i] == "/"))
                    {
                        stack[top+1] = infixarray[i];
                        top++;
                    }
                    else if (top != -1 && (infixarray[i] == "+" || infixarray[i] == "-"))
                    {
                        while (top != -1)
                        {
                            postfixarray[postfixarrayptridx+1] = stack[top];
                            postfixarrayptridx++;
                            top--;
                        }

                        stack[top+1] = infixarray[i];
                        top++;
                    }
                    else if (top != -1 && (infixarray[i] == "x" || infixarray[i] == "/"))
                    {
                        while (top != -1 && stack[top] != "+" && stack[top] != "-")
                        {
                            postfixarray[postfixarrayptridx+1] = stack[top];
                            postfixarrayptridx++;
                            top--;
                        }

                        stack[top+1] = infixarray[i];
                        top++;
                    }
                }

                while (top != -1)
                {
                    postfixarrayptridx++;
                    postfixarray[postfixarrayptridx] = stack[top];
                    top--;
                }

                string answer[infixarraysize+1];
                postfixarrayptridx = 0;
                for (postfixarrayptridx = 0; postfixarrayptridx < infixarraysize; postfixarrayptridx++)
                {
                    if (!(postfixarray[postfixarrayptridx] == "+" || postfixarray[postfixarrayptridx] == "-" || postfixarray[postfixarrayptridx] == "x" || postfixarray[postfixarrayptridx] == "/"))
                    {
                        answer[top+1] = postfixarray[postfixarrayptridx];
                        top++;
                    }
                    else
                    {
                        string rhs;
                        string numstr2 = answer[top];
                        string numstr1 = answer[top-1];
                        // top-=2;

                        if (postfixarray[postfixarrayptridx] == "+")
                        {
                            rhs = addnumstr(numstr1, numstr2);
                        }
                        else if (postfixarray[postfixarrayptridx] == "-")
                        {
                            rhs = subnumstr(numstr1, numstr2);
                        }
                        else if (postfixarray[postfixarrayptridx] == "x")
                        {
                            rhs = mulnumstr(numstr1, numstr2);
                        }
                        else if (postfixarray[postfixarrayptridx] == "/")
                        {
                            rhs = divnumstr(numstr1, numstr2);
                        }

                        // top++;
                        answer[top-1] = rhs;
                        top--;
                    }
                }

                cout << answer[top] << endl;              
                break;
            }
            case 2:
                
                cin >> base >> exponent;
                cout << expostr(base, exponent) << endl;
                break;
            case 3:
                
                cin >> num1 >> num2;
                cout << gcdstr(num1, num2) << endl;
                break;
            case 4:
                
                cin >> num3;
                cout << factostr(num3) << endl;
                break;
        }
    // }

    return 0;
}
