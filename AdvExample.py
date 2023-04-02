class AdvExample():
    """
    Class for representing an adversarial examples
    """
    def __init__(self, img_class, attacked_img):
        """
        Constructor for an adversarial example
        Params:
            img_class: correct image class
            attacked_img: adversarial data (32x32 image)
        """
        self.img_class = img_class
        self.attacked_img = attacked_img