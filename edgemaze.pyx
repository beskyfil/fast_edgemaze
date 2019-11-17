# import queue



import cython
import numpy
cimport numpy
from libcpp.queue cimport queue


class HitError(ValueError):
    pass


class WallHitError(HitError):
    pass


class BorderHitError(HitError):
    pass


IS_TARGET = 1
WALL_LEFT = 2
WALL_UP = 4


def up(numpy.ndarray maze, int x, int y, check_walls=True):
    if x == 0:
        raise BorderHitError
    if check_walls and maze[x, y] & WALL_UP:
        raise WallHitError
    return x - 1, y


def down(numpy.ndarray maze, int x, int y, check_walls=True):
    if x == (maze.shape[0] - 1):
        raise BorderHitError
    if check_walls and maze[x + 1, y] & WALL_UP:
        raise WallHitError
    return x + 1, y


def left(numpy.ndarray maze, int x, int y, check_walls=True):
    if y == 0:
        raise BorderHitError
    if check_walls and maze[x, y] & WALL_LEFT:
        raise WallHitError
    return x, y - 1


def right(numpy.ndarray maze, int x, int y, check_walls=True):
    if y == (maze.shape[1] - 1):
        raise BorderHitError
    if check_walls and maze[x, y + 1] & WALL_LEFT:
        raise WallHitError
    return x, y + 1


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

    nloc = x, y
    while arrows[nloc[0], nloc[1]] != TARGET:
        nloc = DIRS[arrows[nloc]](arrows, nloc[0], nloc[1], check_walls=False)
        path.append(nloc)

    return path


def smallest_dtype(int value):
    for dtype in numpy.int8, numpy.int16, numpy.int32, numpy.int64:
        if dtype(value) == value:
            return dtype
    raise ValueError(f'Maze of size {value} is too big for NumPy to handle')

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

    # dtype = smallest_dtype(maze.size)
    distances = numpy.full((maze.shape[0], maze.shape[1]), -1, dtype=numpy.int64)
    directions = numpy.full((maze.shape[0], maze.shape[1]), UNREACHABLE, dtype=('a', 1))

    # jobs = queue.Queue()
    cdef queue[job] jobs
    cdef job j
    for end in ends(maze):
        directions[end[0], end[1]] = TARGET
        distances[end[0], end[1]] = 0
        # jobs.put((end[0], end[1], 1))
        jobs.push(job(end[0], end[1], 1))

    while not jobs.empty():
        # locx, locy, dist = jobs.get()
        j = jobs.front()
        jobs.pop()
        for walk in [up, left, right, down]:
            try:
                newloc = walk(maze, j.x, j.y)
            except HitError:
                pass
            else:
                # Been there better
                if 0 <= distances[newloc] <= j.dist:
                    continue
                distances[newloc] = j.dist
                directions[newloc] = ANTIDIRS[walk]
                jobs.push(job(newloc[0], newloc[1], j.dist+1))
        
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
