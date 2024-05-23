import numpy as np

arr1= [1,2,3,4,5,6,7,8,9,10]
arr2= [1,2,3,4,5,6,7,8,9,10,11]

# print( int(len(arr1)/2), arr1[int(len(arr1)/2)])
# print( int(len(arr2)/2), arr2[int(len(arr2)/2)])

print( [arr2[int(len(arr2)/2)]] )
print( arr2[int(len(arr2)/2)+1:] )
print( arr2[arr2[int(len(arr2)/2)]-1::-1] )

# print( list(arr2[:int(len(arr2)/2)]).reverse() )
print( list(np.concatenate(   (arr2[arr2[int(len(arr2)/2)]-1::-1], arr2[int(len(arr2)/2)+1:] )     )))