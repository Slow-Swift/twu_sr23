import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


# Crime Data
BUFFER_RADIUS = 20
BUFFER_FALLOFF = 2
DISTANCE_FALLOFF = .8

crimes = [
    (0, 0),
    (30, 0),
    (30, 20)
]

# Animation Data
RESOLUTION = 256
FRAMES = 40
MIN_X, MAX_X = -100, 100
MIN_Y, MAX_Y = -100, 100
a = 1

def value(x, y, crime_x, crime_y, buffer_radius, a):
    distance = betweenDst(abs(x-crime_x), abs(y - crime_y), a)
    buffer_dst = (distance - buffer_radius) / buffer_radius
    falloff = (np.sign(buffer_dst) + 1) * (DISTANCE_FALLOFF + BUFFER_FALLOFF)/2 - BUFFER_FALLOFF
    return 2 * np.exp(-((buffer_dst*falloff)**2)) - 1

def betweenDst(x, y, a):
    return (x**a + y**a) ** (1/a)

def calculateProbabilities(X, Y, a):
    z = 0

    for crime in crimes:
        crime_x, crime_y = crime
        z = value(X, Y, crime_x, crime_y, BUFFER_RADIUS, a) / len(crimes) + z
    return z

def pixelToGrid(pos):
    x, y = pos
    return (
        x/RESOLUTION * (MAX_X - MIN_X) + MIN_X,
        y/RESOLUTION * (MAX_Y - MIN_Y) + MIN_Y
    )

def main():
    X, Y = np.meshgrid(np.linspace(MIN_X, MAX_X, RESOLUTION), np.linspace(MIN_Y, MAX_Y, RESOLUTION))
    z = calculateProbabilities(X, Y, a)

    fig, ax = plt.subplots(layout="constrained")
    ax.set_aspect(1)
    pc = ax.pcolormesh(X, Y, z, vmin=-1, vmax=1, cmap='jet')
    ax.scatter([c[0] for c in crimes], [c[1] for c in crimes], c="green")
    fig.colorbar(pc)
    
    by, bx = pixelToGrid(np.where(z == np.max(z)))
    best = ax.scatter([bx], [by], c="blue")
    
    def update(frame):
        a = -np.cos(2 * np.pi * frame / FRAMES) / 2 + 1.5
        z = calculateProbabilities(X, Y, a)
        by, bx = pixelToGrid(np.where(z == np.max(z)))
        print(by, bx)
        best.set_offsets(np.stack([bx, by]).T)
        pc.update({"array" : z})
        
    ani = anim.FuncAnimation(fig=fig, func=update, frames=FRAMES)
    plt.show()
    
main()