import json

def convert_to_json(data):
    nutrients = [
        'Calories', 'Cal'
        'Total_Fat', 'Fat', 'Total', 'TransFat', 'TotalFat', 'Sat.Fat', 
        'Sat_Fat',
        'Trans_Fat',
        'Cholesterol', 'Cholest.', 'Cholest', 
        'Sodium',
        'Total_Carb', 'Carb.', 'Carbohydrates',
        'Fiber' , 'fiber'
        'Total Sugars', 'Sugars',
        'Added Sugars',
        'Protein',
        'Vit. D',
        'Calcium',
        'Iron',
        'Potassium',
        'Vit.D', 'Vit D'
    ]

    result = {}
    for i in nutrients:
        if i.lower() in data.lower():
            ind = data.index(i)
            quantity = ""
            for kk in data[ind + len(i):]:
                if kk.isdigit():
                    quantity += kk
                else:
                    break
            if quantity:
                result[i] = quantity
            else:
                result[i] = '0'
    # print(type(nutritionData), type(jsonData))
    strData = json.dumps(result)
    return strData

data = "Amountperserving:Calories50,TotalFat3g(4%DV),Sat.Fat0g(2%DV),TransFat0g,Cholest.Omg(0%DV),Sodium170mg(8%Dv),TotalCarb.5g(2%DV),Fiber1g(4%DV),TotalSugars5g(Incl.0gAddedSugars,0%DV),Proteinlessthan1g,Vit.D(6%DV),Calcium(0%DV),Iron(0%DV),Potas.(6%DV)."
out = convert_to_json(data)
print(out)