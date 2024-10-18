# Advance Operating System Assignment - 4

## Peer-to-Peer Group Based File Sharing System

### Tracker :
Compile Tracker : 
`g++ tracker.cpp -o tracker -pthread` 

Run Tracker : 
`./tracker​ <TRACKER INFO FILE> <TRACKER NUMBER>`

Example : 
``./tracker tracker_info.txt 1``


### Client :
Compile Client : 
`g++ -pthread -o client client.cpp -lssl -lcrypto`

Run Client : 
`./client​ <IP>:<PORT> <TRACKER INFO FILE>`

Example : 
``./client 127.0.0.1:8000 tracker_info.txt``


#### tracker_info.txt : 

It contains the details of trackers -
- IP address of both the Trackers 
- Port of both the Trackers 

Example :
``` 
    127.0.0.1
    5000
    127.0.0.1
    6000

```

### Commands :

- Create user account :
```
   create_user​ <user_id> <password> 
```
- Login :
```
    login​ <user_id> <password>
```
- Create Group :
```
    create_group <group_id>
```
- Join Group :
```
    join_group​ <group_id>
```
- Leave Group :
```
    leave_group​ <group_id>
```
- List pending join requests :
```
    list_requests ​<group_id>
```
- Accept Group Joining Request :
```
    accept_request​ <group_id> <user_id>
```
- List All Group In Network :
```
    list_groups
```
- List All sharable Files In Group :
```
    list_files​ <group_id>
```
- Upload File :
```
    upload_file​ <file_path> <group_id​>
```
- Download File :
```
    download_file​ <group_id> <file_name> <destination_path>
```
- Logout :
```
    logout
```
- Show_downloads :
```
    show_downloads
```
- Stop sharing :
```
    stop_share ​<group_id> <file_name>
```





Server (server.cpp)

The server side of the chat application listens for incoming client connections and manages communication between multiple clients. Each client is handled in a separate thread.
Usage:

    Compile the server code using the following command:

    bash

g++ server.cpp -o server

Run the server:

bash

    ./server

    The server will start listening for incoming client connections.

Client (client.cpp)

The client side of the chat application allows a single user to connect to the server and send/receive messages.
Usage:

    Compile the client code using the following command:

    bash

g++ client.cpp -o client

Run the client:

bash

    ./client

    Enter your messages in the client terminal to communicate with the server. Use # to end the connection.

How it Works

    The server listens for incoming client connections on a specified port.
    When a client connects, the server creates a new thread to handle communication with that client.
    Clients can send and receive messages from the server.
    Clients can use # to end the connection.


### Tracker

1. Run Tracker:

```
Format: ./tracker​ <TRACKER INFO FILE> <TRACKER NUMBER>
g++ -pthread -o tracker tracker.cpp; ./tracker tracker_info.txt 1
```

`<TRACKER INFO FILE>` contains the IP, Port details of all the trackers.

```
Ex:
127.0.0.1
2023
127.0.0.1
2024
```

2. Close Tracker:

```
quit
```

### Client:

1. Run Client:

```
Format: ./client​ <IP>:<PORT> <TRACKER INFO FILE>
g++ -pthread -o client client.cpp; ./client 127.0.0.1:8000 ../tracker/tracker_info.txt
```

2. Create user account:

```
create_user​ <user_id> <password>
```

3. Login:

```
login​ <user_id> <password>
```

4. Create Group:

```
create_group​ <group_id>
```

5. Join Group:

```
join_group​ <group_id>
```

6. Leave Group:

```
leave_group​ <group_id>
```

7. List pending requests:

```
list_requests ​<group_id>
```

8. Accept Group Joining Request:

```
accept_request​ <group_id> <user_id>
```

9. List All Group In Network:

```
list_groups
```

10. List All sharable Files In Group:

```
list_files​ <group_id>
```

11. Upload File:

```
​upload_file​ <file_path> <group_id​>
```

12. Download File:​

```
download_file​ <group_id> <file_name> <destination_path>
```

13. Logout:​

```
logout
```

14. Show_downloads: ​(not implemented yet)

```
show_downloads
```

15. Stop sharing: ​

```
stop_share ​<group_id> <file_name>
```
