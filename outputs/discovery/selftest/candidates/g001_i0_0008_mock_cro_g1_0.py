"""Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Coefficients: supply air and heat through the air capacity (floor area); outside wall = facade area x a nominal
2 W/m2K; neighbour exchange through shared walls and doors = contact area x a nominal conductance per m2.  No slab or
wall store.  This is the baseline the other starting models and every LLM candidate must beat.
"""
import numpy as np
HISTORY = 1
REFRESH = 288

def coefficients(geo, hist, U):
    face = U['CELL'] * U['HEIGHT']
    dt = U['DT']
    C_air = U['RHO_C_AIR'] * U['HEIGHT'] * geo['area']
    G_nb = (geo['wall_cells'] * face * 2.5 + geo['door_cells'] * face * 25.0) * (1 - np.eye(geo['V']))
    return {'air': dt / C_air, 'heat': dt / C_air, 'neighbours': dt * G_nb / C_air[:, None], 'outside': dt * geo['facade_faces'] * face * 2.0 / C_air}
