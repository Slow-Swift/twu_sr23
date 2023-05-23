import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from PIL import Image


# Crime Data
BUFFER_RADIUS = 100
BUFFER_FALLOFF = 2
DISTANCE_FALLOFF = .5

crimes = [
    (450, 185),
    (710, 300),
    (555, 485)
]

# Animation Data
RESOLUTION_X, RESOLUTION_Y = 512, 256
FRAMES = 40
MIN_X, MAX_X = 0, 1200
MIN_Y, MAX_Y = 0, 700
a = 1.5

def value(x, y, crime_x, crime_y, buffer_radius, a):
    distance = pNormDst(abs(x-crime_x), abs(y - crime_y), a)
    buffer_dst = (distance - buffer_radius) / buffer_radius
    falloff = (np.sign(buffer_dst) + 1) * (DISTANCE_FALLOFF + BUFFER_FALLOFF)/2 - BUFFER_FALLOFF
    return 2 * np.exp(-((buffer_dst*falloff)**2)) - 1

# Calculate the P-Norm distance
#
# Parameters
#   dx: The x distance
#   dy: The y distance
#
# Returns
#   The P-Norm distance
def pNormDst(dx, dy, p):
    return (dx**p + dy**p) ** (1/p)

def calculateProbabilities(X, Y, a):
    z = 0

    for crime in crimes:
        crime_x, crime_y = crime
        z = value(X, Y, crime_x, crime_y, BUFFER_RADIUS, a) + z
    return z / len(crimes)

def pixelToGrid(pos):
    y, x = pos
    return (
        x/RESOLUTION_X * (MAX_X - MIN_X) + MIN_X,
        y/RESOLUTION_Y * (MAX_Y - MIN_Y) + MIN_Y
    )

def main():
    img = np.asarray(Image.open("Images/Map.png"))
    
    X, Y = np.meshgrid(np.linspace(MIN_X, MAX_X, RESOLUTION_X), np.linspace(MIN_Y, MAX_Y, RESOLUTION_Y))
    z = calculateProbabilities(X, Y, a)

    fig, ax = plt.subplots(layout="constrained")
    ax.imshow(img)
    pc = ax.pcolormesh(X, Y, z, vmin=-1, vmax=1, cmap='jet', alpha=0.5)
    ax.scatter([c[0] for c in crimes], [c[1] for c in crimes], c="red")
    fig.colorbar(pc)
    
    bx, by = pixelToGrid(np.where(z == np.max(z)))
    best = ax.scatter(bx, by, c="green")
    
    def update(frame):
        a = -np.cos(2 * np.pi * frame / FRAMES) / 2 + 1.5
        z = calculateProbabilities(X, Y, a)
        bx, by = pixelToGrid(np.where(z == np.max(z)))
        best.set_offsets(np.stack([bx, by]).T)
        pc.update({"array" : z})
        
    ani = anim.FuncAnimation(fig=fig, func=update, frames=FRAMES)
    plt.show()
    
main()