# MaxHeap List


#-------------------------
from heapq import _heapify_max, _heappop_max, _siftdown_max

class MaxHeap():
    def __init__(self):
        self.arr = []
        self.size=0

    def insert(self, key):
        if not key in self.arr:
            self.arr.append(key)
            _siftdown_max(self.arr, 0, len(self.arr)-1)
            self.size+=1
    def print(self):
        copy=self.arr.copy()
        while len(copy) != 0: # popping items from max_heap
            print(_heappop_max(copy)) # ... unless its empty
            
            
#====
from heapq import _heapify_max, _heappop_max, _siftdown_max

def heappush_max(max_heap, item): 
    max_heap.append(item)
    _siftdown_max(max_heap, 0, len(max_heap)-1)
 
def max_heap(arr):
    copy = arr.copy() # Just maintaining a copy for later use
     
    # Time Complexity = Θ(n); n = no of elements in array
    _heapify_max(arr) # Converts array to max_heap 
     
    # Time Complexity = Θ(logk); k = no of elements in heap
    while len(arr) != 0: # popping items from max_heap
        print(_heappop_max(arr)) # ... unless its empty
         
    arr = copy # since len(arr) = 0
    max_heap = []
    # Time Complexity = Θ(nlogk) - since inserting item one by one 
    for item in arr:
        heappush_max(max_heap, item)
    print("Max Heap is Ready!")
    # Time Complexity = Θ(logk); k = no of elements in heap
    while len(max_heap) != 0: # popping items from max_heap
        print(_heappop_max(max_heap)) # ... unless its empty
     
 
arr = [6,8,9,2,1,5]
max_heap(arr)
# This code is contributed by Swagato Chakraborty (swagatochakraborty123)

print('way2')
arr=[]
for item in [6,8,9,2,1,5]:
    arr.append(item)
    _siftdown_max(arr, 0, len(arr)-1)
while len(arr) != 0: # popping items from max_heap
    print(_heappop_max(arr)) # ... unless its empty

print('way3')
maxhep=MaxHeap()
for item in [6,8,9,2,1,5]:
    maxhep.insert(item)
maxhep.print()
print(maxhep.arr)



print('way4-str')
maxhep2=MaxHeap()

import random
inlist=list('abcdef')
random.shuffle(inlist)
for i in inlist: maxhep2.insert(i)
print(maxhep2.arr)