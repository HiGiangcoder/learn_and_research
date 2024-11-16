list1 = [2, 2, 4, 1, 2, 5, 4, 1]
set1 = set(list1)

print(set1)

set1.add(6)
print(set1)

set1.add((3, 2))
set1.update([3, 2])

print(set1)

set1.remove(2)
print('after remove element 2:', set1)

set1.pop()
print('after using pop() method:', set1)

set1.clear()
print('clear this set:', set1)

print((a := len(set1)))
print(a)
