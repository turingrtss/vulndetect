#!/usr/bin/env python3
"""
Procedural Dungeon Generation - Comparing 4 Algorithms

Implements and compares:
  1. Binary Space Partition (BSP) Trees
  2. Cellular Automata (cave generation)
  3. Random Walk (drunkard's walk)
  4. Room Placement + Corridors

Measures: connectivity, room count, open space ratio, path length distribution.
Outputs: ASCII maps + PNG visualizations.
"""

import random
import json
import time
import numpy as np
from collections import deque


WALL = '#'
FLOOR = '.'
DOOR = '+'
WIDTH = 80
HEIGHT = 40


class DungeonGenerator:
    """Base class for dungeon generators."""

    def __init__(self, width=WIDTH, height=HEIGHT, seed=None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid = [[WALL for _ in range(width)] for _ in range(height)]

    def to_string(self):
        return '\n'.join(''.join(row) for row in self.grid)

    def count_floors(self):
        return sum(row.count(FLOOR) for row in self.grid)

    def open_ratio(self):
        return self.count_floors() / (self.width * self.height)

    def is_connected(self):
        """Check if all floor tiles are reachable from each other."""
        # Find first floor tile
        start = None
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == FLOOR:
                    start = (x, y)
                    break
            if start:
                break

        if not start:
            return False

        # BFS
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and (nx, ny) not in visited
                        and self.grid[ny][nx] in (FLOOR, DOOR)):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == self.count_floors()

    def longest_path(self):
        """BFS from a random floor tile, return max distance."""
        floors = [(x, y) for y in range(self.height)
                  for x in range(self.width) if self.grid[y][x] == FLOOR]
        if not floors:
            return 0

        start = random.choice(floors)
        visited = {start: 0}
        queue = deque([start])
        max_dist = 0

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and (nx, ny) not in visited
                        and self.grid[ny][nx] in (FLOOR, DOOR)):
                    visited[(nx, ny)] = visited[(x, y)] + 1
                    max_dist = max(max_dist, visited[(nx, ny)])
                    queue.append((nx, ny))

        return max_dist

    def count_rooms(self):
        """Count distinct connected components of floor tiles."""
        visited = set()
        rooms = 0

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == FLOOR and (x, y) not in visited:
                    rooms += 1
                    queue = deque([(x, y)])
                    visited.add((x, y))
                    while queue:
                        cx, cy = queue.popleft()
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height
                                    and (nx, ny) not in visited
                                    and self.grid[ny][nx] in (FLOOR, DOOR)):
                                visited.add((nx, ny))
                                queue.append((nx, ny))

        return rooms


class BSPDungeon(DungeonGenerator):
    """Binary Space Partition tree dungeon."""

    def generate(self, min_room=6, max_depth=5):
        rooms = []
        self._split(1, 1, self.width - 2, self.height - 2, 0, max_depth, min_room, rooms)

        # Carve rooms
        for x1, y1, x2, y2 in rooms:
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    self.grid[y][x] = FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            cx1 = (rooms[i][0] + rooms[i][2]) // 2
            cy1 = (rooms[i][1] + rooms[i][3]) // 2
            cx2 = (rooms[i+1][0] + rooms[i+1][2]) // 2
            cy2 = (rooms[i+1][1] + rooms[i+1][3]) // 2
            self._carve_corridor(cx1, cy1, cx2, cy2)

        return self

    def _split(self, x1, y1, x2, y2, depth, max_depth, min_room, rooms):
        w = x2 - x1
        h = y2 - y1

        if depth >= max_depth or (w < min_room * 2 and h < min_room * 2):
            # Create room inside this partition
            rw = random.randint(min_room, max(min_room, w - 2))
            rh = random.randint(min_room, max(min_room, h - 2))
            rx = random.randint(x1 + 1, max(x1 + 1, x2 - rw))
            ry = random.randint(y1 + 1, max(y1 + 1, y2 - rh))
            rooms.append((rx, ry, min(rx + rw, x2 - 1), min(ry + rh, y2 - 1)))
            return

        if w > h:
            # Vertical split
            split = random.randint(x1 + min_room, x2 - min_room)
            self._split(x1, y1, split, y2, depth + 1, max_depth, min_room, rooms)
            self._split(split, y1, x2, y2, depth + 1, max_depth, min_room, rooms)
        else:
            # Horizontal split
            split = random.randint(y1 + min_room, y2 - min_room)
            self._split(x1, y1, x2, split, depth + 1, max_depth, min_room, rooms)
            self._split(x1, split, x2, y2, depth + 1, max_depth, min_room, rooms)

    def _carve_corridor(self, x1, y1, x2, y2):
        x, y = x1, y1
        while x != x2:
            self.grid[y][x] = FLOOR
            x += 1 if x2 > x else -1
        while y != y2:
            self.grid[y][x] = FLOOR
            y += 1 if y2 > y else -1


class CellularAutomata(DungeonGenerator):
    """Cellular automata cave generation."""

    def generate(self, fill_prob=0.45, iterations=5, birth=5, survive=4):
        # Random fill
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if random.random() > fill_prob:
                    self.grid[y][x] = FLOOR

        # Iterate
        for _ in range(iterations):
            new_grid = [[WALL for _ in range(self.width)] for _ in range(self.height)]
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    neighbors = sum(
                        1 for dy in range(-1, 2) for dx in range(-1, 2)
                        if (dx != 0 or dy != 0) and self.grid[y+dy][x+dx] == WALL
                    )
                    if self.grid[y][x] == WALL:
                        new_grid[y][x] = FLOOR if neighbors < birth else WALL
                    else:
                        new_grid[y][x] = FLOOR if neighbors < survive else WALL
            self.grid = new_grid

        return self


class RandomWalk(DungeonGenerator):
    """Drunkard's walk dungeon."""

    def generate(self, target_ratio=0.35, max_steps=50000):
        x = self.width // 2
        y = self.height // 2
        self.grid[y][x] = FLOOR
        steps = 0
        target_floors = int(self.width * self.height * target_ratio)

        while self.count_floors() < target_floors and steps < max_steps:
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nx, ny = x + dx, y + dy
            if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1:
                x, y = nx, ny
                self.grid[y][x] = FLOOR
            steps += 1

        return self


class RoomPlacement(DungeonGenerator):
    """Random room placement with corridor connections."""

    def generate(self, num_rooms=12, min_size=4, max_size=10):
        rooms = []

        for _ in range(num_rooms * 10):  # attempts
            if len(rooms) >= num_rooms:
                break
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            x = random.randint(2, self.width - w - 2)
            y = random.randint(2, self.height - h - 2)

            # Check overlap
            overlap = False
            for rx, ry, rw, rh in rooms:
                if (x < rx + rw + 2 and x + w + 2 > rx and
                        y < ry + rh + 2 and y + h + 2 > ry):
                    overlap = True
                    break

            if not overlap:
                rooms.append((x, y, w, h))
                for cy in range(y, y + h):
                    for cx in range(x, x + w):
                        self.grid[cy][cx] = FLOOR

        # Connect rooms
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in rooms]
        for i in range(len(centers) - 1):
            self._carve_corridor(centers[i][0], centers[i][1],
                                 centers[i+1][0], centers[i+1][1])

        return self

    def _carve_corridor(self, x1, y1, x2, y2):
        x, y = x1, y1
        if random.random() < 0.5:
            while x != x2:
                self.grid[y][x] = FLOOR
                x += 1 if x2 > x else -1
            while y != y2:
                self.grid[y][x] = FLOOR
                y += 1 if y2 > y else -1
        else:
            while y != y2:
                self.grid[y][x] = FLOOR
                y += 1 if y2 > y else -1
            while x != x2:
                self.grid[y][x] = FLOOR
                x += 1 if x2 > x else -1


def evaluate_generator(gen_class, name, runs=20, **kwargs):
    """Run a generator multiple times and collect statistics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    metrics = {
        'open_ratio': [],
        'connected': [],
        'rooms': [],
        'longest_path': [],
        'gen_time': [],
    }

    sample_map = None

    for i in range(runs):
        t0 = time.time()
        gen = gen_class(seed=i)
        gen.generate(**kwargs)
        gen_time = time.time() - t0

        metrics['open_ratio'].append(gen.open_ratio())
        metrics['connected'].append(gen.is_connected())
        metrics['rooms'].append(gen.count_rooms())
        metrics['longest_path'].append(gen.longest_path())
        metrics['gen_time'].append(gen_time)

        if i == 0:
            sample_map = gen.to_string()

    results = {
        'name': name,
        'runs': runs,
        'open_ratio_mean': round(np.mean(metrics['open_ratio']), 3),
        'open_ratio_std': round(np.std(metrics['open_ratio']), 3),
        'connectivity_rate': round(sum(metrics['connected']) / runs, 2),
        'rooms_mean': round(np.mean(metrics['rooms']), 1),
        'rooms_std': round(np.std(metrics['rooms']), 1),
        'longest_path_mean': round(np.mean(metrics['longest_path']), 1),
        'longest_path_std': round(np.std(metrics['longest_path']), 1),
        'gen_time_mean_ms': round(np.mean(metrics['gen_time']) * 1000, 2),
        'sample_map': sample_map,
    }

    print(f"  Open ratio:    {results['open_ratio_mean']:.3f} ± {results['open_ratio_std']:.3f}")
    print(f"  Connectivity:  {results['connectivity_rate']*100:.0f}%")
    print(f"  Rooms:         {results['rooms_mean']:.1f} ± {results['rooms_std']:.1f}")
    print(f"  Longest path:  {results['longest_path_mean']:.0f} ± {results['longest_path_std']:.0f}")
    print(f"  Gen time:      {results['gen_time_mean_ms']:.2f} ms")
    print(f"\n  Sample map (seed=0):")
    print(sample_map)

    return results


def main():
    print("Procedural Dungeon Generation: Algorithm Comparison")
    print(f"Map size: {WIDTH}x{HEIGHT}")
    print(f"Runs per algorithm: 20")

    all_results = {}

    all_results['bsp'] = evaluate_generator(
        BSPDungeon, "BSP Tree", min_room=5, max_depth=4)

    all_results['cellular'] = evaluate_generator(
        CellularAutomata, "Cellular Automata",
        fill_prob=0.45, iterations=5)

    all_results['random_walk'] = evaluate_generator(
        RandomWalk, "Random Walk (Drunkard's Walk)",
        target_ratio=0.35)

    all_results['room_placement'] = evaluate_generator(
        RoomPlacement, "Room Placement + Corridors",
        num_rooms=10, min_size=4, max_size=9)

    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Algorithm':<25} {'Open%':<10} {'Connected':<12} {'Rooms':<10} "
          f"{'Path Len':<12} {'Time (ms)':<10}")
    print("-"*79)

    for name, r in all_results.items():
        print(f"{r['name']:<25} {r['open_ratio_mean']:<10.3f} "
              f"{r['connectivity_rate']*100:<12.0f}% "
              f"{r['rooms_mean']:<10.1f} {r['longest_path_mean']:<12.0f} "
              f"{r['gen_time_mean_ms']:<10.2f}")

    # Save (without maps for JSON)
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'sample_map'}
    save_results['maps'] = {k: v['sample_map'] for k, v in all_results.items()}

    with open('dungeon_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("\nResults saved to dungeon_results.json")


if __name__ == '__main__':
    main()
