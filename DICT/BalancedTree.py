# Python code to convert a sorted array
# to a balanced Binary Search Tree

# binary tree node
import sys

class BTNode:
	def __init__(self, d, parent=None):
		self.item = d
		self.left = None
		self.right = None
		self.parent=parent

# function to convert sorted array to a
# balanced BST
# input : sorted array of integers
# output: root node of balanced BST


class BT():
	def __init__(self, arr):
		self.root = self.sortedArrayToBST(arr)
		self.arr= None

	def sortedArrayToBST(self, arr, parent=None):
		if not arr: return None
		# mid = int( (len(arr))/2 )
		root = BTNode(arr[int( (len(arr))/2 )],parent)
		root.left = self.sortedArrayToBST(arr[:int( (len(arr))/2 )],root)
		root.right = self.sortedArrayToBST(arr[int( (len(arr))/2 )+1:],root)
		return root

	def preOrder(self):
		self.arr=[]
		self.preOrder_helper(self.root)
		arr_c=self.arr.copy()
		self.arr=None
		return arr_c


	def preOrder_helper(self, node):
		if node:
			self.arr.append(node.item)
			self.preOrder_helper(node.left)
			self.preOrder_helper(node.right)

	def print_tree(self):
		self.__print_helper(self.root, "", True)

	# Printing the tree
	def __print_helper(self, node, indent, last):
		if node:
			sys.stdout.write(indent)
			if last:
				sys.stdout.write("R<")
				indent += " "
			else:
				sys.stdout.write("L<")
				indent += "| "

			print(str(node.item) + ">")
			self.__print_helper(node.left, indent, False)
			self.__print_helper(node.right, indent, True)


# driver program to test above function
"""
Constructed balanced BST is 
	4
/ \
2 6
/ \ / \
1 3 5 7
"""

arr = [1, 2, 3, 4, 5, 6, 7]
root = BT(arr)
print(  root.preOrder()  )
root.print_tree()

# This code is contributed by Ishita Tripathi
