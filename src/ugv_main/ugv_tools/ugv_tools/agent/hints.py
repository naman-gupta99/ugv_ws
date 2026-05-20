hints = {
    "easy" : """
    Hint: A column-by-column serpentine sweep keeps x changes to a minimum:
    - If not at (x_min, y_min), first navigate there.
    - For x from x_min to x_max (inclusive):
        - Sweep bottom→top on even columns ((x - x_min) % 2 == 0): move up (y+1) until y == y_max.
        - Sweep top→bottom on odd columns: move down (y-1) until y == y_min.
        - Only after fully sweeping the current column, step right once (x+1) to the next column.
    - Never move right/left in the middle of a column sweep; finish the entire column first
    """,
    "medium" : """
    Hint: A row-by-row serpentine sweep keeps y changes to a minimum
    """,
    "hard" : ""
}