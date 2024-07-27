


#-------------------------
from heapq import _heapify_max, _heappop_max, _siftdown_max

class MinHeap():
    def __init__(self):
        self.arr = []
        heapify(self.arr)
        self.size=0

    def insert(self, key):
        if not key in self.arr:
            heappush(self.arr, key)
            self.size+=1
    def print(self):
        for i in self.arr: 
	        print(i, end = ' ') 
            
            
            
            

# Python3 program to demonstrate working of heapq 

from heapq import heapify, heappush, heappop

# Creating empty heap 
heap = [] 
heapify(heap)

# Adding items to the heap using heappush function 
heappush(heap, 10)
heappush(heap, 30) 
heappush(heap, 20) 
heappush(heap, 400) 

# printing the value of minimum element 
print("Head value of heap : "+str(heap[0])) 

# printing the elements of the heap 
print("The heap elements : ") 
for i in heap: 
	print(i, end = ' ') 
print("\n") 

element = heappop(heap) 

# printing the elements of the heap 
print("The heap elements : ") 
for i in heap: 
	print(i, end = ' ') 
 
 
print('\n----')

sl=MinHeap()
import random
inlist=list('abcdef')
random.shuffle(inlist)
print(inlist)
for i in inlist: sl.insert(i)
print(sl.arr)
sl.print()