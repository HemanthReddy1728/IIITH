CREATE TABLE IF NOT EXISTS shoe_app.users (
    id INT NOT NULL,
    name VARCHAR(80) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    shoes JSON,  -- Store the list of integers as JSON
    PRIMARY KEY (id)
);