list = ['a', 'a', 'b', 'b', 'a', 'a', 'a']
print(list[1:2])
list1 = []
for i in range(0, 6):
    temp = list[i:i+2]
    list1.append(temp)

print(list1)
number = list1.count(['a', 'a'])
index = list1.index(['a', 'a'])
locations =[]
print(number)
for i in range(0, number):
    index = list1.index(['a', 'a'])
    locations.append(index)
    list1[index] = 0
print(locations)
print(list1)
