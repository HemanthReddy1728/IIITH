## Question 1
### Overview
Database Schema:
User:
Username: string
Password: string
ToDo:
Title: string
Description: string
DueDate:Date
Status:Boolean (0 for complete, 1 for not complete)

- Creating Todo app with MERN Stack
- Contains:
1. Register page
2. Login page
3. To do list
4. Create To-do Page(15 Marks):

5. Complete To-do Page(10 Marks):
6. Pending To-Do Page (20 Marks):
7. Logout: (5 marks)


### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

For server:
- Go to server folder
- npm init -y
- npm i
- npm install express mongoose cors nodemon bcrypt jsonwebtoken cookie-parser multer path  
- package.json >> "start": "nodemon server.js" in "scripts"
- Run `npm start` to start the server

For client:
(bootstrap)
npm init vite >> name : client >> React >> Javascript
- Go to client folder
- Run `npm i` to install all the dependencies
- npm install axios react-router-dom 
- Run `npm run dev` to start the client

### Assumption
- None

