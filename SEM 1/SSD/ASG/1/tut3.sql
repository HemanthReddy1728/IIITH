-- CREATE SCHEMA `ssd_lab`;
-- CREATE SCHEMA `ssd_assignment`;

-- Create a database
CREATE DATABASE `SSDTutorial` ;

-- Create Student table
CREATE TABLE SSDTutorial.student (
	student_id INT PRIMARY KEY,
    enrollment_id INT NOT NULL UNIQUE CHECK (enrollment_id > 0),
    student_name VARCHAR(30) NOT NULL,
    student_age INT NOT NULL,
    student_mess VARCHAR(15) DEFAULT 'North',
    student_city VARCHAR(30)
);


-- Create Department table
CREATE TABLE SSDTutorial.department (
	department_id INT PRIMARY KEY,
    department_name VARCHAR(50)
);

-- Create Course table
CREATE TABLE SSDTutorial.course (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(100),
    department_id INT,
    instructor VARCHAR(50)
);


-- Create Register table
CREATE TABLE SSDTutorial.register (
	course_id INT,
    student_id INT
);


-- Create Mess table
CREATE TABLE SSDTutorial.mess (
	mess_id INT PRIMARY KEY,
    mess_name VARCHAR(50)
);


-- Add Foreign key constraint to department_id in course table
ALTER TABLE SSDTutorial.course
ADD FOREIGN KEY (department_id) REFERENCES SSDTutorial.department(department_id);


-- Add Foreign key constraint to course_id in register table
ALTER TABLE SSDTutorial.register
ADD FOREIGN KEY (course_id) REFERENCES SSDTutorial.course(course_id);


-- Add Foreign key constraint to student_id in register table
ALTER TABLE SSDTutorial.register
ADD FOREIGN KEY (student_id) REFERENCES SSDTutorial.student(student_id);


-- Make course_id, student_id as Composite Primary key
ALTER TABLE SSDTutorial.register
ADD PRIMARY KEY (course_id, student_id);


-- ADD register_date column to register
ALTER TABLE SSDTutorial.register
ADD register_date DATE;


-- DELETE (DROP) mess_id column from student
ALTER TABLE SSDTutorial.student
DROP COLUMN student_mess;


-- We can also change data types. Please explore that on your own.


-- DELETE (DROP) mess table
DROP TABLE SSDTutorial.mess;


-- TRUNCATE all tables before we start inserting
TRUNCATE TABLE SSDTutorial.student;
TRUNCATE TABLE SSDTutorial.course;
TRUNCATE TABLE SSDTutorial.department;
TRUNCATE TABLE SSDTutorial.register;



----------------------------------------------------------------


INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (1, 1001, 'Anil', 20, 'Mumbai');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (2, 1002, 'Ajay', 21, 'Agra');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (3, 1003, 'Arun', 18, 'Pune');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (4, 1004, 'Ankita', 20, 'Mumbai');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (5, 1005, 'Ananya', 23, 'Delhi');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (6, 1006, 'Ananya', 25, 'Vizag');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (7, 1007, 'TEST', 25, 'Hyd');
INSERT INTO SSDTutorial.student (student_id, enrollment_id, student_name, student_age, student_city)
VALUES (8, 1008, 'John', 25, 'TEST');


INSERT INTO SSDTutorial.department (department_id, department_name)
VALUES (1, 'Cognitive Science');
INSERT INTO SSDTutorial.department (department_id, department_name)
VALUES (2, 'Software Engg');
INSERT INTO SSDTutorial.department (department_id, department_name)
VALUES (3, 'Computer Vision');


INSERT INTO SSDTutorial.course (course_id, course_name, department_id, instructor)
VALUES (101, 'Introduction to SE', 2, 'Sunil');
INSERT INTO SSDTutorial.course (course_id, course_name, department_id, instructor)
VALUES (102, 'Topics in SE', 2, 'Shaun');
INSERT INTO SSDTutorial.course (course_id, course_name, department_id, instructor)
VALUES (103, 'Introduction to Cognition', 3, 'Shalini'); -- incorrect
INSERT INTO SSDTutorial.course (course_id, course_name, department_id, instructor)
VALUES (104, 'Mind Theory', 1, 'Simran');
INSERT INTO SSDTutorial.course (course_id, course_name, department_id, instructor)
VALUES (105, 'Introduction to ML', 2, 'Mukul');


INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (101, 1, '2023-08-17');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (101, 2, '2023-08-01');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (101, 3, '2023-08-20');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (102, 2, '2023-08-13');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (102, 4, '2023-08-04');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (103, 1, '2023-08-15');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (103, 4, '2023-08-19');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (103, 5, '2023-08-19');
INSERT INTO SSDTutorial.register (course_id, student_id, register_date)
VALUES (104, 5, '2023-08-26');


UPDATE SSDTutorial.course SET department_id=1
WHERE course_id=103;


DELETE FROM SSDTutorial.student
WHERE student_name='TEST' OR student_city='TEST';

-----------------------------------------------------------------

-- View specific columns
SELECT student_id, student_name FROM SSDTutorial.student;
SELECT student_name FROM SSDTutorial.student;

-- DISTINCT
SELECT DISTINCT student_name FROM SSDTutorial.student;

-- WHERE WITH OPERATORS
SELECT * FROM SSDTutorial.student
WHERE student_name = 'Anil';

SELECT * FROM SSDTutorial.student
WHERE student_age > 20;

SELECT * FROM SSDTutorial.student
WHERE student_age <> 20;

-- AND / OR
SELECT * FROM SSDTutorial.student
WHERE student_age >= 21 AND student_city = "Agra";  -- WHERE ((condition1 AND cond2) OR ())

select * from SSDTutorial.student
where student_age >= 20 and student_city in ('Agra', 'Mumbai');


-- LIKE
SELECT * FROM SSDTutorial.student
WHERE student_name LIKE 'A%'; 

SELECT * FROM SSDTutorial.student
WHERE student_name LIKE 'An%';

-- IN
SELECT * FROM SSDTutorial.student
WHERE student_city in ('Mumbai', 'Agra');

-- Aggregate functions
SELECT min(student_age) FROM SSDTutorial.student;
SELECT max(student_age) FROM SSDTutorial.student;
SELECT avg(student_age) FROM SSDTutorial.student;
SELECT sum(student_age) FROM SSDTutorial.student;
SELECT count(*) FROM SSDTutorial.student;

SELECT avg(student_age) FROM SSDTutorial.student
WHERE student_city = 'Mumbai';

-- CLAUSES
-- GROUP BY
SELECT avg(student_age), student_city FROM SSDTutorial.student
WHERE student_city <> 'Vizag'
GROUP BY student_city;

-- ORDER BY
SELECT * FROM SSDTutorial.student
ORDER BY student_name;

SELECT avg(student_age), student_city FROM SSDTutorial.student
WHERE student_city <> 'Vizag'
GROUP BY student_city
ORDER BY student_city desc;

-- HAVING
SELECT avg(student_age), student_city FROM SSDTutorial.student
WHERE student_city <> 'Vizag'
GROUP BY student_city
HAVING avg(student_age) > 20;

-- LIMIT (AS)
SELECT avg(student_age) AS avg_age, student_city FROM SSDTutorial.student
WHERE student_city <> 'Vizag'
GROUP BY student_city
HAVING avg(student_age) > 20
ORDER BY avg_age desc
LIMIT 1;

-- ORDER OF EXECUTION
-- FROM
-- WHERE
-- GROUP BY
-- HAVING
-- SELECT
-- DISTINCT
-- ORDER BY
-- LIMIT

-- HW - CASE Statement
-- HW - String functions (TRIM, SUBSTRING, ETC)


-- SUBQUERY

 -- a subquery (also known as a nested query or inner query) is a query embedded within another query. 
 -- It allows you to use the result of one query as a part of another query. 

-- 1. Find the student name who are enrolled to at least course.

SELECT student_name
FROM SSDTutorial.student
WHERE student_id IN (SELECT student_id FROM SSDTutorial.register);


-- 2. Find the course name that have at least one student enrolled.

SELECT course_name
FROM SSDTutorial.course
WHERE course_id IN (SELECT course_id FROM SSDTutorial.register);


-- 3. List the department name that offer courses with more than 2 students enrolled.

SELECT department_name
FROM SSDTutorial.department
WHERE department_id IN (
    SELECT department_id
    FROM SSDTutorial.course
    WHERE course_id IN (
        SELECT course_id
        FROM SSDTutorial.register
        GROUP BY course_id
        HAVING COUNT(*) > 2
    )
);

-- 4. Retrieve the student names and their ages who are enrolled in courses offered by the 'Software Engg' department.

SELECT student_name, student_age
FROM SSDTutorial.student
WHERE student_id IN (
    SELECT student_id
    FROM SSDTutorial.register
    WHERE course_id IN (
        SELECT course_id
        FROM SSDTutorial.course
        WHERE department_id = (
            SELECT department_id
            FROM SSDTutorial.department
            WHERE department_name = 'Software Engg'
        )
    )
);


-- JOINS

-- INNER JOIN

-- 1. Retrieve the names of students and their enrolled courses.
SELECT student_name, course_name
FROM SSDTutorial.student
INNER JOIN SSDTutorial.register ON student.student_id = register.student_id
INNER JOIN SSDTutorial.course ON register.course_id = course.course_id;

-- 2. List the courses and their instructors offered by the 'Cognitive Science' department.
SELECT course_name, instructor
FROM SSDTutorial.course
INNER JOIN SSDTutorial.department ON course.department_id = department.department_id
WHERE department.department_name = 'Cognitive Science';

-- LEFT JOIN

-- 1. Retrieve a list of all students and their enrolled courses (including those without courses).
SELECT student_name, course_name
FROM SSDTutorial.student
LEFT JOIN SSDTutorial.register ON student.student_id = register.student_id
LEFT JOIN SSDTutorial.course ON register.course_id = course.course_id;

-- RIGHT JOIN

-- 1. Retrieve a list of courses and their enrolled students (including those without students).
SELECT course_name, student_name
FROM SSDTutorial.course
RIGHT JOIN SSDTutorial.register ON course.course_id = register.course_id
RIGHT JOIN SSDTutorial.student ON register.student_id = student.student_id;

-- FULL JOIN

-- 1. List all students and their enrolled courses, showing both enrolled students and courses without students.
SELECT student_name, course_name
FROM SSDTutorial.student
FULL OUTER JOIN SSDTutorial.register ON student.student_id = register.student_id
FULL OUTER JOIN SSDTutorial.course ON register.course_id = course.course_id;

-- SELF JOIN
-- 1. Retrieve pair of students who have registered for the same course.
-- HW


-- SET OPERATIONS
-- UNION, INTERSECTION, etc -- HW
