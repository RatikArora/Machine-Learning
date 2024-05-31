n = 2000000
prime = [True] * (n + 1)
# print(prime)
start = 2
summ = 0

while start*start <=n:
    if prime[start] == True:
        for i in range(start*start,n+1,start):
            prime[i]=False
    start+=1

for i in range(2,n):
    if prime[i]:
        summ += i
    
print(summ)



