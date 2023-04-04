class TargetedAdvExample():
    """
    Class for representing a targeted adversarial example
    """
    def __init__(self, inital_img, attacked_img):
        """
        Constructor for an adversarial example
        Params:
            initial_img: initial data (32x32 image)
            attacked_img: adversarial data (32x32 image)
        """
        self.initial_img = inital_img
        self.attacked_img = attacked_img