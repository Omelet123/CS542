# random generate paper rock scissors
import random
guess_list = ["paper", "rock", "scissors"]
win = [["paper", "rock"], ["rock", "scissors"], ["scissors", "paper"]]

while True:
    computer = random.choice(guess_list)
    people = input('please input：paper, rock, scissors:\n').strip()

    if people not in  guess_list:
        people = input('error! please input：paper, rock, scissors:\n').strip()
        continue
    if computer ==  people:
        print("tie")
        #break
    elif [computer, people] in win:
        print("lose")
        #break
    elif [people, computer] in win:
        print("win")
        break