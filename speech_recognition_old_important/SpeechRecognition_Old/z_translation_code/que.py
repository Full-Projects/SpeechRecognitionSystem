MAX_LENGTH = 2
class queue_se():
    rear = None
    front = None
    length = None
    arr = [None] * MAX_LENGTH
    def __init__(self):
        self.front = 0
        self.rear = MAX_LENGTH - 1
        self.length = 0
    def isEmpty(self):
        return (self.length == 0)
    def isFull(self):
        return (self.length == MAX_LENGTH)
    def addQueue(self, element):
        if self.isFull():
            print("Queue is Full")
        else:
            self.rear = (self.rear + 1) % MAX_LENGTH
            self.arr[self.rear] = element
            self.length += 1
    def deleteQueue(self):
        if self.isEmpty():
            print("queue is Empty")
        else:
            self.rear = (self.rear + 1) % MAX_LENGTH
            self.length -= 1
    def getFrontQueue(self):
        if not self.isEmpty():
            return self.arr[self.front]
    def getRearQueue(self):
        if not self.isEmpty():
            return self.arr[self.rear]
    def printQueue(self):
        if not self.isEmpty():
            i = self.front
            while i != self.rear:
                print(self.arr[i])
                i = (i + 1)%MAX_LENGTH
            print(self.arr[self.rear])
        else:
            print("Print Queue Empty")
    def deleteFrontQueue(self):
        self.deleteQueue()
    def setFrontQueue(self,front):
        self.arr[self.front] = front
    def setRearQueue(self,rear):
        self.arr[self.rear] = rear