import csv

# human is type 0, ai is type 1, random type 2

mapping = []
with open('actual.csv', newline='') as actualfile:
    reader = csv.reader(actualfile, delimiter=',')
    for row in reader:
        temp = {}
        row = row[1:]
        for i in range(len(row)):
            if row[i] == 'human':
                temp[i] = 0
            elif row[i] == 'ai':
                temp[i] = 1
            else:
                temp[i] = 2
        mapping.append(temp)
print(len(mapping))

files = ['Color Design Survey (Responses) - Form Responses 1.csv', 
    'Color Design Survey (Responses) - Form Responses 2.csv', 
    'Color Design Survey (Responses) - Form Responses 3.csv']

curIdx = 0
overall_count = [0, 0, 0]
for f, file in enumerate(files): 
    print("Form responses ", f+1)
    count = [[0 for col in range(3)] for row in range(10)]
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        
        for row in reader:
            row = row[1:]
            for i in range(len(row)):
                count[i][mapping[curIdx+i][int(row[i])-1]] += 1
    for row in range(10):
        for col in range(3):
            overall_count[col] += count[row][col]
            print(count[row][col], end=' ')
        print()
    print()
            
    curIdx += 10

total = 0
for i in range(3):
    print(overall_count[i], end = ' ')
    total += overall_count[i]

for i in range(3):
    print(overall_count[i]/total * 100, end = ' ')
print()