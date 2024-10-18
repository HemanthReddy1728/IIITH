DROP PROCEDURE IF EXISTS DivideTwoNumbers;
DELIMITER $$
CREATE PROCEDURE DivideTwoNumbers(
	IN n1 DECIMAL(10,2),
	IN n2 DECIMAL(10,2),
	OUT result DECIMAL(10,2)
)
BEGIN
	Set result = n1 / n2;
END$$
DELIMITER ;

SET @num1 = 12;
SET @num2 = 7;

CALL DivideTwoNumbers(@num1,@num2,@num3);
SELECT @num3;