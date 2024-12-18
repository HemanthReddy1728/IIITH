USE `CUSTOMER_DB`; 
DROP PROCEDURE IF EXISTS Name_City;
DELIMITER //
CREATE PROCEDURE Name_City(IN amt DECIMAL(10,2))
BEGIN
    SELECT `CUST_NAME`, `CUST_CITY` from `CUSTOMER_DB`.`customer` where `PAYMENT_AMT` > amt;
END;
//
DELIMITER ;

CALL Name_City(5000); 
