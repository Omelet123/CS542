import random


# generate new list for compare
def generate_list(listn, n):
    list0 = []
    for i in range(0, len(listn)-n+1):
        temp = listn[i:i + n]
        list0.append(temp)
    return list0


# find index of the move computer guessed
def find_index(n, listn, compare, number):
    locations = []
    for i in range(0, number-1):
        index = listn.index(compare)
        listn[index] = 0
        location = index + n
        locations.append(location)
    return locations


# find my choice that can beat the computer
def find_choice(index, moveR):
    p = r = s = 0
    for i in range(0, len(index)):
        n = int(index[i])
        if moveR[n] == 'paper':
            p = p+1
        elif moveR[n] == 'rock':
            r = r+1
        else:
            s = s+1
    if p > r and p > s:
        # computer will think I will choose paper, and then it will choose scissors
        return 'rock'
    elif r > p and r > s:
        # computer will think I will choose rock, and then it will choose paper
        return 'scissors'
    elif s > p and s > r:
        # computer will think I will choose scissors, and then it will choose rock
        return 'paper'
    else:
        # computer will randomly generate move
        choice = ["paper", "rock", "scissors"]
        mychoice = random.choice(choice)
        return mychoice


choice = ["paper", "rock", "scissors"]
mymove = []
# randomly generate the first 5 times
for i in range(0, 5):
    mychoice = random.choice(choice)
    mymove.append(mychoice)
# print the first 5 random choice
print(mymove)

# assume we play 100 times of the game
for i in range(5, 100):
    # move base
    compare4 = mymove[i - 4:i]
    compare3 = mymove[i - 3:i]
    compare2 = mymove[i - 2:i]
    compare1 = mymove[i - 1]

    # generate new list for compare
    list4 = generate_list(mymove, 4)
    list3 = generate_list(mymove, 3)
    list2 = generate_list(mymove, 2)
    list1 = generate_list(mymove, 1)

    # find the number of compare base in the new generated list
    number4 = list4.count(compare4)
    number3 = list3.count(compare3)
    number2 = list2.count(compare2)
    number1 = list1.count(compare1)

    # start finding the right choice
    if number4 <= 1 and number3 <= 1 and number2 <= 1:
        index = find_index(1, list1, compare1, number1)
        nextmove = find_choice(index, mymove)
        mymove.append(nextmove)
    elif number4 <= 1 and number3 <= 1:
        index = find_index(2, list2, compare2, number2)
        nextmove = find_choice(index, mymove)
        mymove.append(nextmove)
    elif number4 <= 1:
        index = find_index(3, list3, compare3, number3)
        nextmove = find_choice(index, mymove)
        mymove.append(nextmove)
    else:
        index = find_index(4, list4, compare4, number4)
        nextmove = find_choice(index, mymove)
        mymove.append(nextmove)

# print the list of mymove
print(mymove)
print(len(mymove))




