-- SHOW VARIABLES LIKE "secure_file_priv";
-- Actual value of secure_file_priv -> /var/lib/mysql-files
-- Rename folder containing csv files into 'Question5Artifacts' and copy it into /var/lib/mysql-files folder using sudo cp -ir path/to/Question5Artifacts /var/lib/mysql-files/

DROP TABLE IF EXISTS orderDetails;
DROP TABLE IF EXISTS brandDetails;
DROP TABLE IF EXISTS userDetails;
DROP TABLE IF EXISTS report;

CREATE TABLE brandDetails (
    BrandID INT PRIMARY KEY,
    LaptopBrand VARCHAR(255)
);

CREATE TABLE userDetails (
    UserID INT PRIMARY KEY,
    Name VARCHAR(255),
    FavoriteLaptopBrand VARCHAR(255)
);

CREATE TABLE orderDetails (
    OrderID INT PRIMARY KEY,
    OrderDate DATE,
    BrandID INT,
    BuyerID INT,
    SellerID INT,
    FOREIGN KEY (BrandID) REFERENCES brandDetails(BrandID),
    FOREIGN KEY (BuyerID) REFERENCES userDetails(UserID),
    FOREIGN KEY (SellerID) REFERENCES userDetails(UserID)
);

LOAD DATA INFILE '/var/lib/mysql-files/Question5Artifacts/laptop_brands.csv' INTO TABLE brandDetails FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;
LOAD DATA INFILE '/var/lib/mysql-files/Question5Artifacts/users_table.csv' INTO TABLE userDetails FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;
LOAD DATA INFILE '/var/lib/mysql-files/Question5Artifacts/Orders_table.csv' INTO TABLE orderDetails FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;

CREATE TABLE report (
    UserID INT,
    UserName VARCHAR(255),
    Result VARCHAR(3)
);

INSERT INTO report (UserID, UserName, Result) 
SELECT u.UserID, u.Name AS UserName,
    IF((SELECT COUNT(*) FROM orderDetails o WHERE o.SellerID = u.UserID) > 2 AND (SELECT o.BrandID FROM orderDetails o WHERE o.SellerID = u.UserID ORDER BY o.OrderDate LIMIT 1 OFFSET 1) = (SELECT b.BrandID FROM brandDetails b WHERE b.LaptopBrand = u.FavoriteLaptopBrand), 'Yes', 'No') AS Result
FROM userDetails u;

SELECT * FROM report;