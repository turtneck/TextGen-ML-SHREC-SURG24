# Unsorted List


#-------------------------

class UnsortList():
    def __init__(self):
        self.arr = []
        self.size=0

    def insert(self, key):
        if not key in self.arr:
            self.arr.append(key)
            self.size+=1