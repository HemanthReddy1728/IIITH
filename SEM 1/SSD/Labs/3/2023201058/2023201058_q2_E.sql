-- SUS
SELECT DISTINCT i.user_name, b1.genre FROM issued_users i CROSS JOIN books b1 LEFT JOIN (
    SELECT i2.user_name, b2.genre FROM issued_users i2 JOIN books b2 ON i2.book_id = b2.book_id
) explored ON i.user_name = explored.user_name AND b1.genre = explored.genre WHERE explored.genre IS NULL;