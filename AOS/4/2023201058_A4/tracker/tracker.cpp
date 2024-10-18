#include <arpa/inet.h>
#include <bits/stdc++.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <openssl/sha.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

string logfile, t1ip, ctrackerip, t2ip;

int tcountval = 0, tval = 0;
uint16_t t1port, t2port, ctrackerport;

unordered_map<string, bool> logged_in_already;
unordered_map<string, unordered_map<string, set<string>>> seederList;
vector<string> groupList;
unordered_map<string, string> PieceWiseHash;
unordered_map<string, string> cname_to_port;
unordered_map<string, string> fsize;
unordered_map<string, string> group_admin_list;
unordered_map<string, string> logged_in_people;
unordered_map<string, set<string>> all_pending_group_requests;
unordered_map<string, set<string>> all_group_members;

void synch(int client_socket)
{
    char dummy[5];
    read(client_socket, dummy, 5);
}

void log_write(const string &data)
{
    ofstream log_file(logfile, ios_base::out | ios_base::app);
    log_file << data << endl;
}

void log_creation(char *argv[])
{
    string tracker_number = string(argv[5]);
    ofstream lgfile;
    logfile = "tracker" + tracker_number + "logfile.txt";
    lgfile.open(logfile);
    lgfile.clear();
    lgfile.close();
}

vector<string> getTrackerInfo(char *path)
{
    fstream trackerInfoFile;
    trackerInfoFile.open(path, ios::in);

    string t;

    vector<string> res;

    if (!trackerInfoFile.is_open())
    {
        cout << "Tracker Info file not found.\n";
        exit(-1);
    }

    while (getline(trackerInfoFile, t))
    {
        res.push_back(t);
    }
    trackerInfoFile.close();
    return res;
}

vector<string> string_split(string add, string delimeter)
{
    size_t cposition = 0;
    vector<string> ans;
    while ((cposition = add.find(delimeter)) != string::npos)
    {
        string t = add.substr(0, cposition);
        ans.push_back(t);
        int l = cposition + delimeter.length();
        add.erase(0, l);
    }
    ans.push_back(add);
    return ans;
}

void *check_input(void *arg)
{
    while (true)
    {
        string ip;
        getline(cin, ip);
        if (ip == "quit")
        {
            exit(0);
        }
    }
}

vector<string> string_to_vector(string s)
{
    vector<string> vect;
    stringstream ss(s);
    while (ss >> s)
    {
        vect.push_back(s);
    }
    return vect;
}

void converse_with_client(int client_socket)
{
    log_write("Conversation with the client has started");
    string clientUserID = "", clientGroupID = "";
    while (true)
    {
        char input_line[1024] = {0};
        int read_status = read(client_socket, input_line, 1024);
        string input_linestr = string(input_line);
        if (read_status <= 0)
        {
            logged_in_already[clientUserID] = false;
            close(client_socket);
            break;
        }
        else
        {
            log_write("client request:" + input_linestr);
        }
        vector<string> input_vector = string_to_vector(input_linestr);
        int args_count = input_vector.size();
        if (input_vector[0] == "create_user")
        {
            if (args_count == 3)
            {
                if (logged_in_people.find(input_vector[1]) != logged_in_people.end())
                {
                    write(client_socket, "User has already created account", 32);
                }
                else
                {
                    logged_in_people.insert({input_vector[1], input_vector[2]});
                    write(client_socket, "Account Created", 15);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "login")
        {
            if (args_count == 3)
            {
                int flg = 0;
                if (logged_in_people.find(input_vector[1]) == logged_in_people.end() &&
                    flg == 0)
                {
                    write(client_socket, "Enter Correct UserName/Password", 31);
                    flg = 1;
                }
                if (logged_in_people[input_vector[1]] != input_vector[2] && flg == 0)
                {
                    write(client_socket, "Enter Correct UserName/Password", 31);
                    flg = 1;
                }
                if (logged_in_already[input_vector[1]] == true && flg == 0)
                {
                    write(client_socket, "You are already loggedin", 24);
                    flg = 1;
                }
                if (flg == 0)
                {
                    logged_in_already.insert({input_vector[1], true});
                    clientUserID = input_vector[1];
                    char buffer[96];
                    write(client_socket, "Login Successful", 16);
                    read(client_socket, buffer, 96);
                    string peerAddress = string(buffer);
                    // cout<<"Address stroring while login"<<peerAddress<<endl;
                    cname_to_port[clientUserID] = peerAddress;
                    flg = 1;
                }
            }
            else
            {
                write(client_socket, "Enter correct arguments", 23);
            }
        }
        if (input_vector[0] == "logout")
        {
            logged_in_already[clientUserID] = false;
            write(client_socket, "Logout Successful", 17);
            log_write("logout sucess\n");
        }
        if (input_vector[0] == "create_group")
        {
            if (args_count == 2)
            {
                int flg = 0;
                for (int i = 0; i < groupList.size(); i++)
                {
                    if (groupList[i] == input_vector[1])
                    {
                        write(client_socket, "Group is already present", 24);
                        flg = 1;
                    }
                }
                if (flg == 0)
                {
                    groupList.push_back(input_vector[1]);
                    group_admin_list.insert({input_vector[1], clientUserID});
                    all_group_members[input_vector[1]].insert(clientUserID);
                    clientGroupID = input_vector[1];
                    write(client_socket, "Group created", 13);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "list_groups")
        {
            // cout<<"Grps"<<endl;
            if (args_count == 1)
            {
                write(client_socket, "All groups:", 11);
                synch(client_socket);

                // char dummy[5];
                // read(client_socket, dummy, 5);

                if (groupList.size() == 0)
                {
                    write(client_socket, "No groups found##", 18);
                }
                else
                {

                    string reply = "";
                    for (size_t i = 0; i < groupList.size(); i++)
                    {
                        reply = reply + groupList[i] + "##";
                    }
                    write(client_socket, &reply[0], reply.length());
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "join_group")
        {
            if (args_count == 2)
            {
                if (all_group_members[input_vector[1]].find(clientUserID) !=
                    all_group_members[input_vector[1]].end())
                {

                    if (group_admin_list.find(input_vector[1]) !=
                        group_admin_list.end())
                    {
                        write(client_socket, "You are already in this group", 30);
                    }
                    if (group_admin_list.find(input_vector[1]) ==
                        group_admin_list.end())
                    {
                        write(client_socket, "Invalid group ID.", 18);
                    }
                }
                else
                {
                    all_pending_group_requests[input_vector[1]].insert(clientUserID);
                    write(client_socket, "Group request sent", 18);
                    tval++;
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "leave_group")
        {
            if (args_count == 2)
            {
                int flg = 0;
                write(client_socket, "Leaving group...", 17);
                if (group_admin_list.find(input_vector[1]) == group_admin_list.end() &&
                    flg == 0)
                {
                    write(client_socket, "Can't Leave it is an invalid group ID.", 38);
                    flg = 1;
                }
                if (all_group_members[input_vector[1]].find(clientUserID) ==
                        all_group_members[input_vector[1]].end() &&
                    flg == 0)
                {
                    write(client_socket, "You are not the member of the group", 35);
                    flg = 1;
                }
                if (flg == 0)
                {
                    // char dummy[5];
                    // read(client_socket, dummy, 5);
                    if (group_admin_list[input_vector[1]] == clientUserID)
                    {
                        write(client_socket,
                              "You are the admin of this group, you cant leave!", 48);
                        flg = 1;
                    }
                    if (group_admin_list[input_vector[1]] != clientUserID)
                    {
                        all_group_members[input_vector[1]].erase(clientUserID);
                        write(client_socket, "Group left succesfully", 23);
                        flg = 1;
                    }
                    // write(client_socket, "Admin of the group cannot leave", 31);
                    // flg=1;
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "list_files")
        {
            if (args_count == 2)
            {
                write(client_socket, "All files are:", 15);
                char dummy[5];
                read(client_socket, dummy, 5);

                if (group_admin_list.find(input_vector[1]) != group_admin_list.end())
                {
                    if (seederList[input_vector[1]].size() != 0)
                    {
                        string op = "";
                        for (auto it : seederList[input_vector[1]])
                        {
                            op = op + it.first + "$$";
                        }
                        int l = op.length();
                        op = op.substr(0, l - 2);
                        l = op.length();
                        write(client_socket, &op[0], l);
                    }
                    else
                    {
                        write(client_socket, "No files found.", 15);
                    }
                }
                else
                {
                    write(client_socket, "Enter Valid Group Id", 20);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "list_requests")
        {
            if (args_count == 2)
            {
                write(client_socket, "Fetching group requests...", 27);
                char dummy[5];
                read(client_socket, dummy, 5);
                string response = "";
                // write(client_socket, "Group Requests:",15);

                if (group_admin_list.find(input_vector[1]) !=
                    group_admin_list.end()) //  May be Admin
                {
                    if (group_admin_list[input_vector[1]] == clientUserID) // Is Admin
                    {
                        if (all_pending_group_requests[input_vector[1]].size() != 0)
                        {
                            auto i = all_pending_group_requests[input_vector[1]].begin();
                            while (i != all_pending_group_requests[input_vector[1]].end())
                            {
                                response = response + string(*i) + "$$";
                                i++;
                            }

                            write(client_socket, &response[0], response.length());
                        }
                        else
                        {
                            write(client_socket, "Noreq", 5);
                        }
                    }
                    else // Not Admin
                    {
                        write(client_socket, "Notadmin", 8);
                    }
                }
                else // Not Admin
                {
                    write(client_socket, "Notadmin", 8);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 20);
            }
        }
        if (input_vector[0] == "accept_request") // Showing some problem
        {
            if (args_count == 3)
            {
                write(client_socket, "Accepting request...", 21);
                char dummy[5];
                read(client_socket, dummy, 5);

                if (group_admin_list.find(input_vector[1]) != group_admin_list.end())
                {
                    int flgg = 0;
                    if (group_admin_list.find(input_vector[1])->second == clientUserID)
                    {

                        all_pending_group_requests[input_vector[1]].erase(input_vector[2]);
                        flgg = 1;
                        all_group_members[input_vector[1]].insert(input_vector[2]);

                        write(client_socket, "Request accepted.", 18);
                    }
                    else
                    {
                        write(client_socket, "You are not the admin of this group", 35);
                    }
                }
                else
                {
                    write(client_socket, "Invalid group ID.", 18);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 20);
            }
        }
        if (input_vector[0] == "stop_share")
        {
            if (args_count == 3)
            {
                if (group_admin_list.find(input_vector[1]) != group_admin_list.end())
                {
                    if (seederList[input_vector[1]].find(input_vector[2]) !=
                        seederList[input_vector[1]].end())
                    {
                        seederList[input_vector[1]][input_vector[2]].erase(clientUserID);
                        tcountval++;
                        if (seederList[input_vector[1]][input_vector[2]].size() != 0)
                        {
                            write(client_socket, "Stopped sharing the file", 25);
                        }
                        else
                        {
                            seederList[input_vector[1]].erase(input_vector[2]);
                            write(client_socket, "Stopped sharing the file", 25);
                        }
                    }
                    else
                    {
                        write(client_socket, "File not yet shared in the group", 32);
                    }
                }
                else
                {
                    write(client_socket, "Invalid group ID.", 18);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "show_downloads")
        {
            write(client_socket, "Loading...", 10);
        }
        if (input_vector[0] == "upload_file")
        {
            if (args_count == 3)
            {
                if (all_group_members.find(input_vector[2]) !=
                    all_group_members.end())
                {
                    if (all_group_members[input_vector[2]].find(clientUserID) !=
                        all_group_members[input_vector[2]].end())
                    {
                        struct stat buffer;
                        const string &s = input_vector[1];
                        if ((stat(s.c_str(), &buffer) == 0))
                        {
                            char fdetails[524288] = {0};
                            write(client_socket, "Uploading...", 12);

                            if (read(client_socket, fdetails, 524288))
                            {
                                string fdetailsstr = string(fdetails);
                                if (fdetailsstr != "error")
                                {
                                    string hshval_of_pieces = "";
                                    vector<string> fdet = string_split(fdetailsstr, "$$");
                                    // fdet = [filepath, peer address, file size, file hash,
                                    // piecewise hash]
                                    vector<string> tvect = string_split(string(fdet[0]), "/");
                                    string filename = tvect.back();

                                    size_t i = 4;
                                    while (i < fdet.size())
                                    {
                                        hshval_of_pieces = hshval_of_pieces + fdet[i];
                                        if (i == fdet.size() - 1)
                                        {
                                            i++;
                                            continue;
                                        }
                                        else
                                        {
                                            hshval_of_pieces = hshval_of_pieces + "$$";
                                            i++;
                                        }
                                    }
                                    PieceWiseHash[filename] = hshval_of_pieces;

                                    if (seederList[input_vector[2]].find(filename) ==
                                        seederList[input_vector[2]].end())
                                    {
                                        tcountval++;
                                        // cout<<"Equal "<<clientUserID<<endl;
                                        seederList[input_vector[2]].insert(
                                            {filename, {clientUserID}});
                                        fsize[filename] = fdet[2];
                                    }
                                    else
                                    {
                                        // cout<<"NOT Equal "<<clientUserID<<endl;

                                        seederList[input_vector[2]][filename].insert(clientUserID);
                                        fsize[filename] = fdet[2];
                                    }

                                    write(client_socket, "Uploaded", 8);
                                }
                            }
                        }
                        else
                        {
                            write(client_socket, "DirectFileNotFound", 18);
                        }
                    }
                    else
                    {
                        write(client_socket, "NotAMember", 10);
                    }
                }
                else
                {
                    write(client_socket, "GroupNotFound", 13);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 19);
            }
        }
        if (input_vector[0] == "download_file")
        {
            if (args_count == 4)
            {
                if (all_group_members.find(input_vector[1]) !=
                    all_group_members.end())
                {
                    if (all_group_members[input_vector[1]].find(clientUserID) !=
                        all_group_members[input_vector[1]].end())
                    {
                        const string &s = input_vector[3];
                        struct stat buffer;
                        if (stat(s.c_str(), &buffer) == 0)
                        {
                            char fileDetails[524288] = {0};
                            write(client_socket, "Downloading...", 13);
                            if (read(client_socket, fileDetails, 524288))
                            {
                                string fdetailsstr = string(fileDetails);
                                string reply = "";
                                vector<string> fdet = string_split(string(fileDetails), "$$");
                                if (seederList[input_vector[1]].find(fdet[0]) ==
                                    seederList[input_vector[1]].end())
                                {
                                    tcountval++;
                                    write(client_socket, "File not found", 14);
                                }
                                else
                                {
                                    for (auto i : seederList[input_vector[1]][fdet[0]])
                                    {
                                        // cout<<"fjsgrs"<<endl;
                                        //  if(!logged_in_already[i])
                                        //  {
                                        //      continue;
                                        //  }
                                        // cout<<"Value of i accessing for reply "<<i<<endl;
                                        reply += cname_to_port[i] + "$$";
                                    }
                                    // cout<<"Reply Sending while downloading"<<reply<<endl;
                                    reply += fsize[fdet[0]];
                                    // writeLog("seeder list: " + reply);
                                    write(client_socket, &reply[0], reply.length());
                                    synch(client_socket);
                                    // char dum[5];
                                    // read(client_socket, dum, 5);

                                    write(client_socket, &PieceWiseHash[fdet[0]][0],
                                          PieceWiseHash[fdet[0]].length());

                                    tval++;
                                    seederList[input_vector[1]][input_vector[2]].insert(
                                        clientUserID);
                                }
                                tval++;
                            }
                        }
                        else
                        {
                            write(client_socket, "DirectoryNotPresent", 20);
                        }
                    }
                    else
                    {
                        tcountval++;
                        write(client_socket, "notGroupMember", 14);
                    }
                }
                else
                {
                    write(client_socket, "noSuchGroup", 11);
                }
            }
            else
            {
                write(client_socket, "Enter Valid Command", 20);
            }
        }
    }
    string client_socketstr = to_string(client_socket);
    log_write("pthread ended successfully for socket number " + client_socketstr);
    close(client_socket);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Pass the correct number of arguments" << endl;
        return -1;
    }

    log_creation(argv);

    string arg2str = string(argv[2]);

    vector<string> trackeraddress = getTrackerInfo(argv[1]);

    if (arg2str == "1")
    {
        t1ip = trackeraddress[0];
        t1port = stoi(trackeraddress[1]);
        ctrackerip = t1ip;
        ctrackerport = t1port;
    }
    else if (string(argv[2]) == "2")
    {
        t2ip = trackeraddress[2];
        t2port = stoi(trackeraddress[3]);
        ctrackerip = t2ip;
        ctrackerport = t2port;
    }
    else
    {
        cout << "More than 2 trackers are not supported" << endl;
        return -1;
    }

    log_write("Tracker 1 Address : " + string(t1ip) + ":" + to_string(t1port));
    log_write("Tracker 2 Address : " + string(t2ip) + ":" + to_string(t2port));
    log_write("Log file name : " + logfile + "\n");

    int tsock = socket(AF_INET, SOCK_STREAM, 0);

    pthread_t exitDetectionThreadId;

    if (tsock == 0)
    {
        cout << "Socket Failed" << endl;
        exit(1);
    }
    else
    {
        log_write("Socket of tracker is created.");
    }
    int opt = 1;
    if (setsockopt(tsock, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt)))
    {
        perror("setsockopt");
        exit(1);
    }
    else
    {
        log_write("Set sock opt is done atleast");
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_port = htons(ctrackerport);

    int inetstatus = inet_pton(AF_INET, &ctrackerip[0], &address.sin_addr);
    if (inetstatus <= 0)
    {
        cout << "Address is wrong INET error" << endl;
        return -1;
    }
    else
    {
        log_write("inet_pton is done");
    }
    int bindstatus = bind(tsock, (struct sockaddr *)&address, sizeof(address));
    if (bindstatus < 0)
    {
        cout << "Binding Failed" << endl;
        return -1;
    }
    else
    {
        log_write("Binding Completed");
    }
    int lstatus = listen(tsock, 3);
    if (lstatus < 0)
    {
        cout << "Listening failed" << endl;
        return -1;
    }
    else
    {
        log_write("Listening");
    }
    vector<thread> threadsvect;

    if (pthread_create(&exitDetectionThreadId, NULL, check_input, NULL) == -1)
    {
        perror("pthread");
        exit(EXIT_FAILURE);
    }
    else
    {
        log_write("P thread working successfully");
    }
    int addresslen = sizeof(address);
    while (true)
    {
        int client_socket =
            accept(tsock, (struct sockaddr *)&address, (socklen_t *)&addresslen);
        if (client_socket == 0)
        {
            cout << "Connection not accepted" << endl;
        }
        else
        {
            log_write("Connection accepted");
        }
        threadsvect.push_back(thread(converse_with_client, client_socket));
    }
    auto it = threadsvect.begin();
    while (it != threadsvect.end())
    {
        if (!(it->joinable()))
        {
            it++;
            continue;
        }
        it->join();
        it++;
    }
    log_write("EXITING.");
    return 0;
}