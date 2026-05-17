class Node:
    def __init__(
        self,
        attributeIndex=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.value = value

        self.attributeIndex = attributeIndex
        self.threshold = threshold

        self.left = left
        self.right = right