n = 2000000
arr = [True]* (n + 1)
s = 2
while s*s <=n:
    if arr[s] == True:
        for i in range(s*s,n+1,s):
            arr[s] = False
    s+=1

summ = 0
for i in range(2,n):
    if arr[i]:
        summ+=i

print(summ)



import threading

def print_numbers():
    for i in range(5):
        print(i)

def print_letters():
    for letter in 'ABCDE':
        print(letter)

