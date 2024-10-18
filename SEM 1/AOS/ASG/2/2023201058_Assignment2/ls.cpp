#include "headers.h"
using namespace std;

// Function to resolve user name from UID
string get_username(uid_t uid) 
{
    struct passwd *pwd = getpwuid(uid);
    if (pwd != nullptr) 
    {
        return pwd->pw_name;
    }
    return to_string(uid);
}

// Function to resolve group name from GID
string get_groupname(gid_t gid) 
{
    struct group *grp = getgrgid(gid);
    if (grp != nullptr) 
    {
        return grp->gr_name;
    }
    return to_string(gid);
}

// Function to list files and directories in long format
void list_files_long_format(const string& path) 
{
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) != -1) 
    {
        cout << (S_ISDIR(file_stat.st_mode) ? "d" : "-");
        cout << ((file_stat.st_mode & S_IRUSR) ? "r" : "-");
        cout << ((file_stat.st_mode & S_IWUSR) ? "w" : "-");
        cout << ((file_stat.st_mode & S_IXUSR) ? "x" : "-");
        cout << ((file_stat.st_mode & S_IRGRP) ? "r" : "-");
        cout << ((file_stat.st_mode & S_IWGRP) ? "w" : "-");
        cout << ((file_stat.st_mode & S_IXGRP) ? "x" : "-");
        cout << ((file_stat.st_mode & S_IROTH) ? "r" : "-");
        cout << ((file_stat.st_mode & S_IWOTH) ? "w" : "-");
        cout << ((file_stat.st_mode & S_IXOTH) ? "x" : "-");
        cout << " " << setw(2) << file_stat.st_nlink;
        cout << " " << setw(8) << get_username(file_stat.st_uid);
        cout << " " << setw(8) << get_groupname(file_stat.st_gid);
        cout << " " << setw(8) << file_stat.st_size;

        struct tm *timeinfo;
        timeinfo = localtime(&file_stat.st_mtime);

        char date_buffer[80];
        strftime(date_buffer, sizeof(date_buffer), "%b %d %Y %H:%M", timeinfo);

        cout << " " << date_buffer;
        if (path[0] == '.' and path[1] == '/')
        {
            cout << " " << path.substr(2) << endl;
        }
        else
        {
            cout << " " << path << endl;
        }
        
    } 
    
    else 
    {
        perror("stat");
    }
}


// Function to list files and directories
void list_files(const vector<string>& paths, bool show_hidden, bool long_format) 
{
    for (const string& path : paths) 
    {
        string current_path = path;
        if (path == "~") 
        {
            // Expand the tilde character to the user's home directory
            wordexp_t p;
            if (wordexp("~", &p, 0) == 0 && p.we_wordc > 0) 
            {
                current_path = string(p.we_wordv[0]);
                wordfree(&p);
            } else {
                cerr << "Unable to determine home directory" << endl;
                continue;
            }
        }

        DIR *dir = opendir(current_path.c_str());
        if (dir == nullptr) 
        {
            // Handle files or invalid paths
            if (errno == ENOTDIR) 
            {
                if (long_format) 
                {
                    list_files_long_format(current_path);
                } else {
                    cout << current_path << " ";
                }
            } 
            
            else 
            {
                perror("opendir");
            }
        } 

        else 
        {
            // Handle directories
            struct dirent *entry;
            while ((entry = readdir(dir)) != nullptr) 
            {
                if (!show_hidden && entry->d_name[0] == '.') 
                {
                    continue; // Skip hidden files
                }
                string full_path = current_path + "/" + entry->d_name;
                if (long_format) 
                {
                    list_files_long_format(full_path);
                } 
                
                else 
                {
                    cout << entry->d_name << " ";
                }
            }
            closedir(dir);
        }
    }
    cout << endl;
}