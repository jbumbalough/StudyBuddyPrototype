# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class Topic:
    def __init__(self, name):
        self.name = name
        self.materials = []

    def add_material(self, new_material):
        self.materials.append(new_material)

    def show(self):
        print("For Topic: ", self.name)
        print("You should study ", self.materials)


Greedy = Topic('CS 210 Greedy Approach')
Greedy.add_material('Lesson 7 Slides')
Greedy.add_material('Foundations of Algorithms 5th ed. Chapter 4')

#for x in range(len(Greedy.materials)):
#   print(Greedy.materials[x]),

#print(Greedy.name)

Greedy.show()


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
