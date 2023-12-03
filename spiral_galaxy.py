# Create a 3-D simulation of a spiral galaxy.

import math
from random import randint, uniform, random
import numpy as np
import matplotlib.pyplot as plt
from window import Window


plt.style.use('dark_background')

# Set the radius of the galactic disc (scaling factor):



def build_spiral(b, r, rot_fac, fuz_fac, arm):
    """Build a spiral arm for tkinter display with Logarithmic spiral formula.

    b = growth factor: negative = spiral to left; larger = more open
    r = radius
    rot_fac = rotation factor for spiral arm
    fuz_fac = fuzz factor to randomly shift star positions
    arm = spiral arm (0 = leading edge, 1 = trailing stars)
    """
    spiral_stars = []
    fuzz = int(0.030 * abs(r))  # Scalable initial amount to shift locations
    for i in range(0, 800, 2):  # Use range(520) for central "hole"
        theta = math.radians(-i)
        x = r * math.exp(b * theta) * math.cos(theta - math.pi * rot_fac) \
            - randint(-fuzz, fuzz) * fuz_fac
        y = r * math.exp(b * theta) * math.sin(theta - math.pi * rot_fac) \
            - randint(-fuzz, fuzz) * fuz_fac
        spiral_stars.append([x, y])

    return np.array(spiral_stars)


b = 0.3
SCALE = 350  # Use range of 200 - 700.
fuz_fac = 1.5
temp = build_spiral(b=b, r=SCALE, rot_fac=2, fuz_fac=fuz_fac, arm=2)
temp2 = build_spiral(b=b, r=SCALE, rot_fac=1, fuz_fac=fuz_fac, arm=1)
temp3 = build_spiral(b=b, r=SCALE, rot_fac=0.5, fuz_fac=fuz_fac, arm=0)

plt.scatter(temp[:, 0], temp[:, 1], c='w', marker='.', s=5)
plt.scatter(temp2[:, 0], temp2[:, 1], c='w', marker='.', s=5)
plt.scatter(temp3[:, 0], temp3[:, 1], c='w', marker='.', s=5)


# ax.scatter(*zip(*trailing_arm), c='w', marker='.', s=2)
# ax.scatter(*zip(*core_stars), c='w', marker='.', s=1)
# ax.scatter(*zip(*inner_haze_stars), c='w', marker='.', s=1)
# ax.scatter(*zip(*outer_haze_stars), c='lightgrey', marker='.', s=1)
plt.show()