#include <iostream>
#include <vector>

using namespace std;

class TrieNode
{
public:
    TrieNode *children[26];
    bool isEndOfWord;
    int index;

    TrieNode()
    {
        for (int i = 0; i < 26; i++)
        {
            children[i] = nullptr;
        }
        isEndOfWord = false;
        index = -1;
    }
};

class Trie
{
public:
    TrieNode *root;

    Trie()
    {
        root = new TrieNode();
    }

    void insertWord(string &word)
    {
        TrieNode *currentNode = root;

        for (int i = 0; i < word.length(); i++)
        {
            if (currentNode->children[word[i] - 'a'] == nullptr)
            {
                currentNode->children[word[i] - 'a'] = new TrieNode();
            }
            currentNode = currentNode->children[word[i] - 'a'];
        }
        currentNode->isEndOfWord = true;
    }

    bool searchWord(string &word)
    {
        TrieNode *currentNode = root;

        for (int i = 0; i < word.size(); i++)
        {
            if (currentNode->children[word[i] - 'a'] == nullptr)
            {
                return false;
            }
            else
            {
                currentNode = currentNode->children[word[i] - 'a'];
            }
        }
        if (currentNode->isEndOfWord)
        {
            return true;
        }
        return false;
    }

    vector<string> autoComplete(string &word)
    {
        vector<string> WordsList;
        TrieNode *currentNode = root;
        string currentWord;

        for (int i = 0; i < word.size(); i++)
        {
            if (currentNode->children[word[i] - 'a'] == nullptr)
            {
                return WordsList;
            }
            else
            {
                currentWord.push_back(word[i]);
                currentNode = currentNode->children[word[i] - 'a'];
            }
        }

        TrieNode *tempNode = currentNode;

        if (tempNode->isEndOfWord)
        {
            WordsList.push_back(currentWord);
        }

        for (int i = 0; i < 26; i++)
        {
            if (tempNode->children[i] != nullptr)
            {
                currentWord.push_back(i + 'a');
                getAutoCompleteWordsList(tempNode->children[i], WordsList, currentWord);
                currentWord.pop_back();
            }
        }

        return WordsList;
    }

    vector<string> autoCorrect(string &word)
    {
        vector<int> currentRow(word.size() + 1);
        vector<string> results;
        string currentWord;

        for (int i = 0; i < currentRow.size(); i++)
        {
            currentRow[i] = i;
        }

        for (int i = 0; i < 26; i++)
        {
            if (root->children[i] != nullptr)
            {
                currentWord.push_back(i + 'a');
                autoCorrectWordsList(root->children[i], i + 'a', word, currentRow, results, currentWord);
                currentWord.pop_back();
            }
        }

        return results;
    }

    void getAutoCompleteWordsList(TrieNode *node, vector<string> &WordsList, string currentWord)
    {
        if (node->isEndOfWord)
        {
            WordsList.push_back(currentWord);
        }

        for (int i = 0; i < 26; i++)
        {
            if (node->children[i] != nullptr)
            {
                currentWord.push_back(i + 'a');
                getAutoCompleteWordsList(node->children[i], WordsList, currentWord);
                currentWord.pop_back();
            }
        }
    }

    void autoCorrectWordsList(TrieNode *node, char character, string &word, vector<int> &previousRow, vector<string> &results, string &currentWord)
    {
        vector<int> currentRow;
        currentRow.push_back(previousRow[0] + 1);

        for (int i = 1; i <= word.size(); i++)
        {
            int insertOperation = currentRow[i - 1] + 1, deleteOperation = previousRow[i] + 1, replaceOperation = 0;
            if (word[i - 1] != character)
            {
                replaceOperation = previousRow[i - 1] + 1;
            }
            else
            {
                replaceOperation = previousRow[i - 1];
            }

            currentRow.push_back(min(min(insertOperation, deleteOperation), replaceOperation));
        }

        if (currentRow[word.size()] < 4 && node->isEndOfWord)
        {
            results.push_back(currentWord);
        }

        int currentMinEntry = 2147483647;
        for (auto x : currentRow)
            currentMinEntry = min(x, currentMinEntry);

        if (currentMinEntry < 4)
        {
            for (int i = 0; i < 26; i++)
            {
                if (node->children[i] != nullptr)
                {
                    currentWord.push_back(i + 'a');
                    autoCorrectWordsList(node->children[i], i + 'a', word, currentRow, results, currentWord);
                    currentWord.pop_back();
                }
            }
        }
    }
};

int main()
{
    int numWords, numQueries;
    cin >> numWords >> numQueries;

    Trie trie;
    for (int i = 0; i < numWords; i++)
    {
        string word;
        cin >> word;
        trie.insertWord(word);
    }

    while (numQueries--)
    {
        int queryType;
        string word;
        cin >> queryType >> word;

        if (queryType == 1)
        {
            cout << trie.searchWord(word) << endl;
        }
        else if (queryType == 2)
        {
            vector<string> WordsList = trie.autoComplete(word);
            cout << WordsList.size() << endl;
            for (int i = 0; i < WordsList.size(); i++)
            {
                cout << WordsList[i] << endl;
            }
        }
        else if (queryType == 3)
        {
            vector<string> WordsList = trie.autoCorrect(word);
            cout << WordsList.size() << endl;
            for (int i = 0; i < WordsList.size(); i++)
            {
                cout << WordsList[i] << endl;
            }
        }
    }

    return 0;
}
