foodList = []
foodCalories = []

with open('foodCalories.txt') as foodCal:
    for fileLine in foodCal:
        foodCalories.append(fileLine)

with open('foodNames.txt') as foodName:
    for fileLine in foodName:
        foodList.append(fileLine)

print(foodCalories[0])
print(foodList[0])
