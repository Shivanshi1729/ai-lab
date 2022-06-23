graph = {
    'S': { 'A':4, 'B':5},
    'A': { 'S':4, 'E':15, 'B': 3, 'D':8},
    'B': { 'S':5, 'A':3, 'D':6, 'C':4}, 
    'C': { 'B': 4, 'D': 4, 'G': 7},
    'D': { 'A': 8, 'G': 2, 'E': 2, 'B': 6, 'C': 4},
    'E': { 'A': 15, 'D': 2},
    'G': { 'C':7, 'D': 2}
}

from queue import PriorityQueue
import copy

def branch_bound(graph, source, goal):
    pq = PriorityQueue()
        
    bound = {}
    for i in graph:
        bound[i] = 1000000
    
    initial = (0, source, [])
    pq.put(initial)
    
    answer = []
    while not pq.empty():
        d, node, path = pq.get()
        print('visiting node: ' + str(node) + ' ' + str(d), end=' ')
        print(path)
        if bound[node] > d:
            bound[node] = d
            # explore this
            print('adding nodes: ')
            for i in graph[node]:
                newpath = copy.deepcopy(path)
                newpath.append(node)
                newnode = (d + graph[node][i], i, newpath)
                print('\t' + str(newnode))
                pq.put(newnode)                    
            if node == goal:
                # goal found 
                # terminate all path with length greater than currently found
                # update bounds for rest of nodes
                for i in bound:
                    if bound[i] > d:
                        bound[i] = d
                answer = path
        else:
            print("\tbetter path exist for " + str(node) + " of length " + str(bound[node]))

    if answer == []:
        print("no path!")
    else:
        print('\nPath found: ' + '->'.join(answer) + '->' + goal) 

branch_bound(graph, 'S', 'G')

