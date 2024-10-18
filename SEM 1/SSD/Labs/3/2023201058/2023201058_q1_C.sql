-- SELECT * FROM SSDLab.menu;
-- update SSDLab.menu set price = price - 5 where food_type = 'non-veg';
-- update SSDLab.menu set price = price - 1 where food_type = 'veg';
update SSDLab.menu set price = (
	case
		when food_type = 'non-veg' then price - 5
		when food_type = 'veg' then price - 1
	end
);		
SELECT * FROM SSDLab.menu;