import itertools
import pandas as pd

alphabet="123456789FGJLNRSWYZ"

all_combinations = itertools.product(alphabet, repeat=8)

df = pd.DataFrame(columns=list('ABCDEFGH'))  # ABCDEFGHI

j=0
number_iter=0
for row in all_combinations:
    k=1
    
    print("\r {0} - {1}".format(row,number_iter), end='')    
    
    # print ("----------------Our row is -",row)
    # looking for how many times the same element is repeated
    counter = {}
    l=list(row)
    for elem in l:
        counter[elem] = counter.get(elem, 0) + 1
    doubles = {element: count for element, count in counter.items() if count > 1}
    # print("Element: repeated ", doubles)
    ll=list(doubles.values())
    # print ("ll =========== ",ll)
    # if more than 4 - stop searching
    for i in ll: 
        if i>=5:
            # print ("...more than 4")
            k=0
            break
    # looking for 3 identical elements together
    for i1 in range(len(row)-2):
        if row[i1] == row[i1+1] == row[i1+2]:
            # print ("...more than 3 are the same one by one")
            k=0
            break
    if k==1:
        # print (l)
        df.loc[0]=l
        
        df.to_csv('filtered_pass.csv',mode='a', header=False)
    
        #j+=1
        
    number_iter+=1
    # if number_iter==2000000: break



# df.to_csv('pass_testing.csv') 


"""
# shuffle ramdom lines in file
import random
filename='pass_testing.csv'
f=open(filename)
lines = f.readlines()
f.close()
random.shuffle(lines)
f=open(filename,'w')
f.writelines(lines)
f.close()
"""