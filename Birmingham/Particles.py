import random
import time
import os

ROOM_WIDTH = 50
ROOM_HEIGHT = 30

WINDOW_ROW = ROOM_HEIGHT // 2

room_grid = [[0 for _ in range(ROOM_WIDTH)] for _ in range(ROOM_HEIGHT)]

def inject_heat_particles(count):
    """Inject heat particles into the room through the window."""
    for _ in range(count):
        room_grid[WINDOW_ROW][0] += 1  # Inject at the leftmost column

def move_particle(x, y):
    """Move particle randomly to one of the neighboring cells."""
    direction = random.choice(['up', 'down', 'left', 'right'])
    if direction == 'up' and y > 0:
        y -= 1
    elif direction == 'down' and y < ROOM_HEIGHT - 1:
        y += 1
    elif direction == 'left' and x > 0:
        x -= 1
    elif direction == 'right' and x < ROOM_WIDTH - 1:
        x += 1
    return x, y

def simulate_step():
    """Move all particles in the room."""
    new_grid = [[0 for _ in range(ROOM_WIDTH)] for _ in range(ROOM_HEIGHT)]

    for y in range(ROOM_HEIGHT):
        for x in range(ROOM_WIDTH):
            for _ in range(room_grid[y][x]):
                new_x, new_y = move_particle(x, y)
                new_grid[new_y][new_x] += 1

    return new_grid

def display_room():
    """Visualize the current state of the room."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
    for y in range(ROOM_HEIGHT):
        line = ""
        for x in range(ROOM_WIDTH):
            if room_grid[y][x] > 0:
                # Heat particles are shown as symbols (more particles = more intense)
                if room_grid[y][x] > 4:
                    line += "█"
                elif room_grid[y][x] > 2:
                    line += "▓"
                else:
                    line += "▒"
            else:
                line += "."
        print(line)
    print("\n")

def main():
    steps = 1000                 # Total simulation steps
    particles_per_step = 5       # Heat particles entering per step
    delay = 0.1                  # Time between frames (seconds)

    for step in range(steps):
        inject_heat_particles(particles_per_step)
        global room_grid
        room_grid = simulate_step()
        display_room()
        print(f"Step {step + 1}/{steps}")
        time.sleep(delay)

    print("Simulation complete.")

if __name__ == "__main__":
    main()
