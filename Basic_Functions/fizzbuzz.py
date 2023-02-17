"""
fizzbuzz

Write a python script which prints the numbers from 1 to 100,
but for multiples of 3 print "fizz" instead of the number,
for multiples of 5 print "buzz" instead of the number,
and for multiples of both 3 and 5 print "fizzbuzz" instead of the number.
"""
import numpy as np


nlist = np.arange(1,101) #Creates an array with the list of numbers 1 to 100

for number in nlist: 
#Check each element in the array

    if number % 3 == 0 and number % 5 == 0:
        print('fizzbuzz')
        #If the number mod 3 and mod 5 is 0 then it prints 'fizzbuzz'
        #This is added as the first condition as it overrides the following two conditions being true separately
        
    elif number % 3 == 0:
        print('fizz')
        #If the number is only divisible by 3, print 'fizz'
        
    elif number % 5 == 0:
        print('buzz')
        #Else if the number is only divisible by 5, print 'buzz'
        
    else:
        print(number)
