
from sys import getsizeof
import pandas as pd

allVar = dir()

memTable = pd.DataFrame(columns = ['Variable','Size'])
for var in allVar:
    mem = getsizeof(var)
    memTable = memTable.append({'Variable':var,'Size':mem}, ignore_index = True)

memTable = memTable.sort_values(by = 'Size', ascending = False)

print('Total memory usage is ' + str(round(memTable['Size'].sum()/1000000,3)) + 'MB')

def analyzeMemory(allVar):
    allVar
    memTable = pd.DataFrame(columns = ['Variable','Size'])
    for var in allVar:
        mem = getsizeof(eval(var))
        memTable = memTable.append({'Variable':var,'Size':mem}, ignore_index = True)
    memTable = memTable.sort_values(by = 'Size', ascending = False)
    print('Total memory usage is ' + str(round(memTable['Size'].sum()/1000000,3)) + 'MB')
    return(memTable)


analyzeMemory(dir())

a = 1000

eval('a')
