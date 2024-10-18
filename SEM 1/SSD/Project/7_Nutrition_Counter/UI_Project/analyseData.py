import json
import matplotlib.pyplot as plt


def processData():
    nutritionData =[]

    f = open("output.txt", "r")
    strData = ""
    try:
        for i in f.readlines():
            l = i.split()
            if l:
                nutritionData.append(l)
                for data in l:
                    strData += data
                strData += " "
        convert_to_json(strData)

    except:
        print("Data was not extracted properly. Please upload a clear picture.")


def convert_to_json(data):
    nutrients =  [['Calories', 'Cal', 'Energy'],
        ['Fat', 'Total_Fat', 'TotalFat'], 
        ['Saturated Fat', 'Sat.Fat', 'Sat_Fat'],
        ['Trans_Fat', 'Trans Fat'],
        ['Cholesterol', 'Cholest.', 'Cholest'],
        ['Sodium'],
        ['Carbohydrate', 'Total_Carb', 'Carb.'],
        ['Fiber', 'Fibre'],
        ['Sugars', 'sugar'],
        ['Protein'],
        ['Vit. D', 'Vitamin D', 'Vit D'],
        ['Calcium'],
        ['Iron'],
        ['Potassium']]

    result = {}
    for k in nutrients:
        for i in k:
            if i.lower() in data.lower():
                ind = data.lower().index(i.lower())
                quantity = ""
                endInd = ind + len(i)
                for kk in data[ind + len(i):]:
                    if kk.isdigit() or kk == '.':
                        endInd += 1
                        quantity += kk
                    else:
                        break
                # UNIT
                unit = ""
                unitsList = []
                while(endInd < len(data) and data[endInd].isalpha()):
                    unit += data[endInd]
                    endInd += 1
                if quantity:
                    if quantity[-1] == '9':
                        quantity = quantity[:len(quantity)]
                        unit = "g"
                    result[k[0]] = quantity
                    unitsList.append(unit)
                else:
                    result[k[0]] = "0"
                    unitsList.append('g')
                break
    finalOutput = {}
    printData = {}
    for k,v in result.items():
        if float(v):
            printData[k] = float(v)
        if k in ['Fat', 'Protein', 'Carbohydrate', 'Sugars']:
            finalOutput[k] = float(v)
    with open('nutri.json', 'w') as file:
        json.dump(finalOutput, file, indent=2)
    print(result)
    display(finalOutput)
    return result

def display(data):
    labels = data.keys()
    sizes = data.values()

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Nutritional Composition')

    plt.savefig('static/pie_chart.png')
    plt.axis('equal')

if __name__ == "__main__":
    processData()
 

