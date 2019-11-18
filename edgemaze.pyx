import cython
import numpy
cimport numpy
from libcpp.queue cimport queue

IS_TARGET = 1
WALL_LEFT = 2
WALL_UP = 4

cdef struct loc:
    int x
    int y

cpdef loc up(numpy.ndarray maze, int x, int y, check_walls=True):
    if x == 0:
        return loc(-10, -10)
    if check_walls and maze[x, y] & WALL_UP:
        return loc(-10, -10)
    return loc(x - 1, y)


cpdef loc down(numpy.ndarray maze, int x, int y, check_walls=True):
    if x == (maze.shape[0] - 1):
        return loc(-10, -10)
    if check_walls and maze[x + 1, y] & WALL_UP:
        return loc(-10, -10)
    return loc(x + 1, y)


cpdef loc left(numpy.ndarray maze, int x, int y, check_walls=True):
    if y == 0:
        return loc(-10, -10)
    if check_walls and maze[x, y] & WALL_LEFT:
        return loc(-10, -10)
    return loc(x, y - 1)


cpdef loc right(numpy.ndarray maze, int x, int y, check_walls=True):
    if y == (maze.shape[1] - 1):
        return loc(-10, -10)
    if check_walls and maze[x, y + 1] & WALL_LEFT:
        return loc(-10, -10)
    return loc(x, y + 1)


def ends(numpy.ndarray maze):
    return numpy.asarray(numpy.where(maze & IS_TARGET)).T


DIRS = {
    b'^': up,
    b'<': left,
    b'>': right,
    b'v': down,
}

ANTIDIRS = {
    down: b'^',
    right: b'<',
    left: b'>',
    up: b'v'
}

UNREACHABLE = b' '
TARGET = b'X'

def arrows_to_path(numpy.ndarray arrows, int x, int y):
    if arrows[x, y] == UNREACHABLE:
        raise ValueError('Cannot construct path for unreachable cell')
    path = [(x, y)]
    cdef loc nloc

    nloc = loc(x, y)
    while arrows[nloc.x, nloc.y] != TARGET:
        nloc = DIRS[arrows[nloc.x, nloc.y]](arrows, nloc.x, nloc.y, check_walls=False)
        path.append((nloc.x, nloc.y))

    return path

cdef struct job:
    int x
    int y
    int dist

@cython.boundscheck(False)
@cython.wraparound(False)
def flood(numpy.ndarray maze):
    if maze.ndim != 2 or not numpy.issubdtype(maze.dtype, numpy.integer):
        raise TypeError('maze must be a 2-dimensional array of integers')

    # Initialize everything as unreachable
    cdef numpy.ndarray[numpy.int64_t, ndim=2] distances
    cdef numpy.ndarray[numpy.int8_t, ndim=2] directions
    cdef int locx, locy

    distances = numpy.full((maze.shape[0], maze.shape[1]), -1, dtype=numpy.int64)
    directions = numpy.full((maze.shape[0], maze.shape[1]), UNREACHABLE, dtype=('a', 1))

    cdef queue[job] jobs
    cdef job j
    cdef loc newloc
    for end in ends(maze):
        directions[end[0], end[1]] = TARGET
        distances[end[0], end[1]] = 0
        jobs.push(job(end[0], end[1], 1))

    while not jobs.empty():
        j = jobs.front()
        jobs.pop()
        for walk in [up, left, right, down]:
            newloc = walk(maze, j.x, j.y)
            if newloc.x == -10 and newloc.y == -10:
                pass
            else:
                # Been there better
                if 0 <= distances[newloc.x, newloc.y] <= j.dist:
                    continue
                distances[newloc.x, newloc.y] = j.dist
                directions[int(newloc.x), int(newloc.y)] = ANTIDIRS[walk]
                jobs.push(job(newloc.x, newloc.y, j.dist+1))
        
    return distances, directions


def is_reachable(directions):
    return UNREACHABLE not in directions


class AnalyzedMaze:
    def __init__(self, maze):
        cdef numpy.ndarray[numpy.int64_t, ndim=2] distances
        cdef numpy.ndarray[numpy.int8_t, ndim=2] directions
        self.maze = maze
        self.distances, self.directions = flood(maze)
        self.is_reachable = is_reachable(self.directions)

    def path(self, column, row):
        return arrows_to_path(self.directions, column, row)


def analyze(maze):
    return AnalyzedMaze(maze)
