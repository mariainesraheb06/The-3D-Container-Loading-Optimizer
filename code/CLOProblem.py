
@dataclass                          #python will directly generate its constructor __init__
class CLOProblem :
    Container: container
    se_boxes: List[Box]
    #def __init__(self, container, seq_boxes):
     #   self.container = container
      #  self.seq_boxes = seq_boxes

    def __post_init__(self):
        """
        To check wether :
                        every box do fit in the container
                        all boxes might fit
        """
        total_box_volume=0
        for box in  self.seq_boxes:
            total_boxes_volume += box.get_volume()
            if not fits_in_container(box, self.container):
                raise ValueError(f"Box {box.get_dims()} can not fit in"
                                f"Container {self.container.get_dims()} even with rotations!")
        container_volume = self.container.get_volume()
        if total_boxes_volume > container.volume :
            print(  f"Warning: Total box volume ({total_box_volume}) exceeds "
                    f"container volume ({container_volume})")
        self._total_boxes_volume = total_boxes_volume
        self._container_volume = container_volume

    @staticmethod
    def fits_in_container(box, container):
        box_dims = sorted(box.get_dims())
        container_dims = sorted(container.get_dims())
        return all(box_dims[k]<=container_dims[i] for i in range(3))

    def get_difficulty(self) -> str:           #additional !
        """To estimate the problem difficulty"""
        ratio = self.total_box_volume / self.container_volume
        if ratio < 0.5:
            return "Easy"
        elif ratio < 0.8:
            return "Medium"
        else:
            return "Hard"

    def display(self) -> str :
        return f"""
                    Problem : {self.get_difficulty()}
                    Container: {self.container.get_dims()}
                    Boxes : {len(self.seq_boxes)}
                    Total box volume: {self.total_box_volume:.2f}
                    Container volume: {self.container_volume:.2f}
                """
