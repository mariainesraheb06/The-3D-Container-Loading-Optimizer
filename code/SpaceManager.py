class SpaceManager:
    def init(self, container_space):
        # Start with one big empty container
        self.spaces = [container_space]

    # ===============================
    # Find best space for a box
    # ===============================
    def find_best_placement(self, box):
        best_score = None
        best_space = None
        best_dims = None

        for space in self.spaces:
            fits, dims = space.get_best_fit(box)

            if fits:
                l, w, h = dims

                # Heuristic: minimize leftover edges
                score = (space.l - l) + (space.w - w) + (space.h - h)

                if best_score is None or score < best_score:
                    best_score = score
                    best_space = space
                    best_dims = dims

        return best_space, best_dims

    # ===============================
    # Place a box and update spaces
    # ===============================
    def place_box(self, space, dims):
        if space not in self.spaces:
            return

        # Remove used space
        self.spaces.remove(space)

        # Split into new spaces
        new_spaces = space.split(*dims)

        # Add and clean
        self.spaces.extend(new_spaces)
        self.clean_spaces()

    # ===============================
    # Clean useless spaces
    # ===============================
    def clean_spaces(self):
        cleaned = []

        for s in self.spaces:
            # Remove zero or negative spaces
            if s.l <= 0 or s.w <= 0 or s.h <= 0:
                continue

            # Remove contained spaces
            if any(other != s and self.contains(other, s) for other in self.spaces):
                continue

            cleaned.append(s)

        self.spaces = cleaned

    # ===============================
    # Geometry helpers
    # ===============================
    def contains(self, a, b):
        return (
            a.x <= b.x and
            a.y <= b.y and
            a.z <= b.z and
            a.x + a.l >= b.x + b.l and
            a.y + a.w >= b.y + b.w and
            a.z + a.h >= b.z + b.h
        )

    # ===============================
    # Optional: debug
    # ===============================
    def print_spaces(self):
        for s in self.spaces:
            print(s)