"""
Egyptian algorithm
"""

def egyptian_multiplication(a, n):
    """
    returns the product a * n

    assume n is a nonegative integer
    """
    def isodd(n):
        """
        returns True if n is odd
        """
        return n & 0x1 == 1

    if n == 1:
        return a
    if n == 0:
        return 0

    if isodd(n):
        return egyptian_multiplication(a + a, n // 2) + a
    else:
        return egyptian_multiplication(a + a, n // 2)


#if __name__ == '__main__':
#    # this code runs when executed as a script
#    for a in [1,2,3]:
#        for n in [1,2,5,10]:
#            print("{} * {} = {}".format(a, n, egyptian_multiplication(a,n)))


def power(a, n):
    """
    computes the power a ** n

    assume n is a nonegative integer
    
    This function calls on the egyptian_multiplication function to multiply 'a' by itself (as in a**2) then multiplies that result
    with the remaining powers of 'a' (by recursively calling on power() but subtracting 2 from n until the end conditions of n=0 or 1
    are reached)
    """
    
    if n == 1:
        return a
    if n == 0:
        return 1
   
    return egyptian_multiplication(a,a) * power(a, n - 2)
   
    
    
if __name__ == '__main__':
    # this code runs when executed as a script
    for a in [2,3,4,5]:
        for n in [3,4]:
            print("{} ** {} = {}".format(a, n, power(a,n)))