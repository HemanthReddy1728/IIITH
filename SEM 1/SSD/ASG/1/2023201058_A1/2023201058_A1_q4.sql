-- SHOW VARIABLES LIKE "secure_file_priv";
-- Actual value of secure_file_priv -> /var/lib/mysql-files
-- Rename folder containing csv files into 'Question4Artifacts' and copy it into /var/lib/mysql-files folder using sudo cp -ir path/to/Question4Artifacts /var/lib/mysql-files/

DROP TABLE IF EXISTS CourseCompletions;
DROP TABLE IF EXISTS CertificateCompletions;
DROP TABLE IF EXISTS Employee;
DROP TABLE IF EXISTS Courses;
DROP TABLE IF EXISTS Certifications;
DROP TABLE IF EXISTS UpdatedEmpSkills;
DROP PROCEDURE IF EXISTS UpdateEmployeeSkills;

CREATE TABLE Employee (
    emp_no INT PRIMARY KEY,
    emp_name VARCHAR(100),
    emp_joining_date DATE,
    emp_salary DECIMAL(10, 2)
);

CREATE TABLE Courses (
    course_id VARCHAR(100) PRIMARY KEY,
    course_name VARCHAR(100)
);

CREATE TABLE Certifications (
    cert_id VARCHAR(100) PRIMARY KEY,
    cert_name VARCHAR(100)
);

CREATE TABLE CourseCompletions (
    course_id VARCHAR(100),
    emp_no INT,
    date_completed DATE,
    FOREIGN KEY (course_id) REFERENCES Courses(course_id),
    FOREIGN KEY (emp_no) REFERENCES Employee(emp_no)
);

CREATE TABLE CertificateCompletions (
    cert_id VARCHAR(100),
    emp_no INT,
    date_completed DATE,
    FOREIGN KEY (cert_id) REFERENCES Certifications(cert_id),
    FOREIGN KEY (emp_no) REFERENCES Employee(emp_no)
);

LOAD DATA INFILE '/var/lib/mysql-files/Question4Artifacts/Employee.csv' INTO TABLE Employee FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES (emp_no, emp_name, @joining_date, emp_salary) SET emp_joining_date = STR_TO_DATE(@joining_date, '%d-%m-%Y');
LOAD DATA INFILE '/var/lib/mysql-files/Question4Artifacts/Courses.csv' INTO TABLE Courses FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;
LOAD DATA INFILE '/var/lib/mysql-files/Question4Artifacts/Certifications.csv' INTO TABLE Certifications FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;
LOAD DATA INFILE '/var/lib/mysql-files/Question4Artifacts/CourseCompletions.csv' INTO TABLE CourseCompletions FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES (course_id, emp_no, @date_completed) SET date_completed = STR_TO_DATE(@date_completed, '%d-%m-%Y');
LOAD DATA INFILE '/var/lib/mysql-files/Question4Artifacts/CertificateCompletions.csv' INTO TABLE CertificateCompletions FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES (cert_id, emp_no, @date_completed) SET date_completed = STR_TO_DATE(@date_completed, '%d-%m-%Y');

-- UPDATE Employee SET emp_joining_date = STR_TO_DATE(@joining_date, '%d-%m-%Y');
-- UPDATE CourseCompletions SET date_completed = STR_TO_DATE(@date_completed, '%d-%m-%Y');
-- UPDATE CertificateCompletions SET date_completed = STR_TO_DATE(@date_completed, '%d-%m-%Y');

DELIMITER //
CREATE PROCEDURE UpdateEmployeeSkills()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE emp_no INT;
    DECLARE emp_salary DECIMAL(10, 2);
    DECLARE skill_level VARCHAR(20);
    DECLARE increment_percentage INT;
    DECLARE new_salary DECIMAL(10, 2);
	DECLARE CourseCompletionsCount INT;
	DECLARE CertificateCompletionsCount INT;
    DECLARE emp_joining_date DATE;
    DECLARE cur CURSOR FOR
        -- Select relevant employee data
        SELECT e.emp_no, e.emp_salary, e.emp_joining_date
        FROM Employee e;
        -- WHERE e.emp_joining_date <= '2022-07-31'; -- Filter employees who joined before the given date
        
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    -- Create UpdatedEmpSkills table
    DROP TABLE IF EXISTS UpdatedEmpSkills;
    CREATE TABLE UpdatedEmpSkills (
        emp_no INT,
        curr_salary DECIMAL(10, 2),
        increment INT,
        new_salary DECIMAL(10, 2),
        skill_level VARCHAR(20)
    );

    OPEN cur;
    read_loop: LOOP
        FETCH cur INTO emp_no, emp_salary, emp_joining_date;
        IF done THEN
            LEAVE read_loop;
        END IF;

        -- Calculate skill level 
        -- SET CourseCompletionsCount = (SELECT COUNT(*) FROM CourseCompletions WHERE emp_no = emp_no);
        -- SET CertificateCompletionsCount = (SELECT COUNT(*) FROM CertificateCompletions WHERE emp_no = emp_no);

        SELECT COUNT(*) INTO CourseCompletionsCount FROM CourseCompletions WHERE CourseCompletions.emp_no = emp_no;
        SELECT COUNT(*) INTO CertificateCompletionsCount FROM CertificateCompletions WHERE CertificateCompletions.emp_no = emp_no;
        
        SET skill_level = CASE
            WHEN CourseCompletionsCount >= 2 AND CertificateCompletionsCount >= 2  AND emp_joining_date <= '2022-07-31' THEN
                CASE
                    WHEN CourseCompletionsCount >= 10 AND CertificateCompletionsCount >= 8 THEN 'Expert'
                    WHEN CourseCompletionsCount >= 6 AND CertificateCompletionsCount >= 6 THEN 'Advanced'
                    WHEN CourseCompletionsCount >= 4 AND CertificateCompletionsCount >= 4 THEN 'Intermediate'
                    ELSE 'Beginner'
                END
            ELSE NULL
        END;

		-- Calculate increment %
        SET increment_percentage = CASE
            WHEN skill_level = 'Expert' THEN 25
            WHEN skill_level = 'Advanced' THEN 20
            WHEN skill_level = 'Intermediate' THEN 15
            WHEN skill_level = 'Beginner' THEN 10
            ELSE 0
        END;

        -- IF skill_level IS NOT NULL THEN
		SET new_salary = emp_salary * (1 + increment_percentage / 100);
		-- Insert data into UpdatedEmpSkills table
		INSERT INTO UpdatedEmpSkills (emp_no, curr_salary, increment, new_salary, skill_level) VALUES (emp_no, emp_salary, increment_percentage, new_salary, skill_level);
        -- END IF;
    END LOOP;
    CLOSE cur;
END;
//
DELIMITER ;

-- Call the stored procedure to update the UpdatedEmpSkills table
CALL UpdateEmployeeSkills();

SELECT * FROM UpdatedEmpSkills;