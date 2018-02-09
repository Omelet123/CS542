# random generate paper rock scissors
import random
choice = ["paper", "rock", "scissors"]
mymove = []
for i in range(0,100):
    mychoice = random.choice(choice)
    mymove.append(mychoice)
print(mymove)
#print(len(mymove))
