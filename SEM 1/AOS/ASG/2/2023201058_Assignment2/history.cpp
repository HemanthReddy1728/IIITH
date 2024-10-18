#include "headers.h"
using namespace std;

class History {
public:
    History();
    void addCommand(const string& command);
    vector<string> getHistory() const;
    vector<string> getHistory(int num) const;
    string getCommand(int index) const;
    string getLastCommand() const;
    string getPreviousCommand();
    string getNextCommand();
    bool isEmpty() const;
    int size() const;
    int getMaxSize() const;

private:
    vector<string> history_;
    int maxSize_;
    int currentHistoryIndex_;
};

History::History() : maxSize_(20) {}

void History::addCommand(const string& command) {
    history_.push_back(command);
    if (history_.size() > maxSize_) {
        history_.erase(history_.begin());
    }
}

vector<string> History::getHistory() const {
    return history_;
}

vector<string> History::getHistory(int num) const {
    int start = max(static_cast<int>(history_.size()) - num, 0);
    int end = history_.size();
    return vector<string>(history_.begin() + start, history_.begin() + end);
}

string History::getCommand(int index) const {
    if (index >= 0 && index < history_.size()) {
        return history_[index];
    }
    return "";
}

int History::size() const {
    return history_.size();
}

int History::getMaxSize() const {
    return maxSize_;
}

string History::getLastCommand() const {
    if (!history_.empty()) {
        return history_.back();
    }
    return "";
}

string History::getPreviousCommand() {
    if (currentHistoryIndex_ > 0) {
        currentHistoryIndex_--;
        return history_[currentHistoryIndex_];
    }
    return getLastCommand();
}

string History::getNextCommand() {
    if (currentHistoryIndex_ < history_.size() - 1) {
        currentHistoryIndex_++;
        return history_[currentHistoryIndex_];
    }
    return getLastCommand();
}

bool History::isEmpty() const {
    return history_.empty();
}
