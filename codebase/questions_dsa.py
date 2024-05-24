# write a program to rotate a image by 90 degrees

image = [
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]
]

# we will get the transpose of given matrix
# 

def rotate_image_90_degrees(matrix):
    n = len(matrix)  # Get the length of the matrix (number of rows)

    # Transpose the matrix
    for i in range(n):
        for j in range(i + 1, n):  # Iterate only the upper triangle to avoid redundant swaps
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]  # Swap elements to get the transpose

# Define the image matrix
image = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Call the function to rotate the image by 90 degrees (transpose it first)
rotate_image_90_degrees(image)
print(image)  # Print the transposed matrix

n = len(image)  # Get the length of the image matrix
# Reverse each row for anticlockwise rotation
for i in range(n):
    image[i].reverse()  # Reverse each row to complete the 90 degrees rotation




###########################################################################################
###########################################################################################

# Write python program for linked list implementation with following functionalities

# append - add new element at the end of existing list
# prepend - add new element at the beginning
        
# def append_ele(list_1,element):
#     # length_of_list = len(list_1)
#     list_1 = list_1+[element]
#     return list_1

# Node class to represent an element in the linked list
class Node:
    def __init__(self, data):
        self.data = data  # Data stored in the node
        self.next = None  # Pointer to the next node, initially set to None

# LinkedList class to handle operations on the linked list
class LinkedList:
    def __init__(self):
        self.head = None  # Initialize the head of the linked list to None

    # Append a new node with the given data at the end of the list
    def append(self, data):
        new_node = Node(data)  # Create a new node
        if not self.head:  # If the list is empty, set the new node as the head
            self.head = new_node
            return
        last = self.head  # Start from the head
        while last.next:  # Traverse to the last node
            last = last.next
        last.next = new_node  # Set the next of the last node to the new node

    # Prepend a new node with the given data at the beginning of the list
    def prepend(self, data):
        new_node = Node(data)  # Create a new node
        new_node.next = self.head  # Link the new node to the current head
        self.head = new_node  # Update the head to the new node

    # Display the linked list elements
    def display(self):
        elements = []  # List to hold the elements of the linked list
        current = self.head  # Start from the head
        while current:  # Traverse through the list
            elements.append(current.data)  # Append the data of each node to the elements list
            current = current.next  # Move to the next node
        print("Linked List:", elements)  # Print the elements of the linked list

# Example usage:
ll = LinkedList()  # Create a new linked list
ll.append(10)  # Append 10 to the list
ll.append(20)  # Append 20 to the list
ll.prepend(5)  # Prepend 5 to the list
ll.display()  # Display the linked list
