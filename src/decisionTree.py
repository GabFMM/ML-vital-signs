from sample import Sample
from node import Node

from abc import ABC, abstractmethod
import random

class DecisionTree(ABC):
    def __init__(self):
        self.root = None

    @abstractmethod
    def testTree(self, testSamples: list[Sample]) -> float:
        pass

    @abstractmethod
    def hasToStop(self, samples: list[Sample]) -> bool:
        pass

    @abstractmethod
    def majorityOutput(self, samples: list[Sample]) -> float | int:
        # used in recursiveBuildTree
        # returns float if the tree is regressor
        # returns int if the tree is classifier
        pass

    @abstractmethod
    def calculateAuxMeasure(self, trainingSamples: list[Sample]) -> float | int:
        # used in chooseAttribute
        # can return information gain or variance, depends to implementation
        pass

    @abstractmethod
    def calculateMeasure(self, leftOutputs: list, rightOutputs: list, aux) -> float | int:
        # used in chooseAttribute to calculateMeasure
        # it is an optional auxiliar method
        pass

    def split(
        self, samples: list[Sample], attributeIndex: int, threshold: float
        ) -> tuple[list[Sample], list[Sample]]:

        left = []
        right = []

        for sample in samples:

            value = sample.attributesList[attributeIndex]

            if value <= threshold:
                left.append(sample)

            else:
                right.append(sample)

        return left, right

    def chooseAttribute(self, trainingSamples: list[Sample], numAttributes: int | None = None) -> tuple[int, float]:
        if numAttributes == None:
            numAttributes = len(trainingSamples[0].attributesList)

        bestAttrIndex = -1
        bestThreshold = 0.0
        bestMeasure = float("-inf")

        aux = self.calculateAuxMeasure(trainingSamples)

        # all samples have the same number of attributes
        for currentAttrIndex in random.sample(range(len(trainingSamples[0].attributesList)), k=numAttributes):
            attrValues = [
                sample.attributesList[currentAttrIndex] 
                for sample in trainingSamples
            ]

            # avoid repeated threshold values
            attrValues = sorted(set(attrValues))

            for i in range(len(attrValues) - 1):
                threshold = (attrValues[i] + attrValues[i + 1]) / 2

                left, right = self.split(trainingSamples, currentAttrIndex, threshold)

                if len(left) == 0 or len(right) == 0:
                    continue

                leftOutputs = [sample.output for sample in left]
                rightOutputs = [sample.output for sample in right]

                measure = self.calculateMeasure(leftOutputs, rightOutputs, aux)

                if  measure > bestMeasure:
                    bestMeasure = measure
                    bestThreshold = threshold
                    bestAttrIndex = currentAttrIndex

        return bestAttrIndex, bestThreshold

    def recursiveBuildTree(self, trainingSamples: list[Sample], numAttributes: int | None = None) -> Node:
        if self.hasToStop(trainingSamples):
            return Node(value=self.majorityOutput(trainingSamples))
    
        bestAttrIndex, bestThreshold = self.chooseAttribute(trainingSamples, numAttributes)

        if bestAttrIndex == -1:
            return Node(value=self.majorityOutput(trainingSamples))

        left, right = self.split(trainingSamples, bestAttrIndex, bestThreshold)

        if len(left) == 0 or len(right) == 0:
            return Node(value=self.majorityOutput(trainingSamples))        

        leftChild = self.recursiveBuildTree(left, numAttributes)

        rightChild = self.recursiveBuildTree(right, numAttributes)

        return Node(
            attributeIndex=bestAttrIndex,
            threshold=bestThreshold,
            left=leftChild,
            right=rightChild
        )

    def buildTree(self, trainingSamples: list[Sample], numAttributes: int | None = None):
        self.root = self.recursiveBuildTree(trainingSamples, numAttributes)

    def recursivePredict(self, node, sample: Sample):
        # leaf
        if node.value is not None:
            return node.value

        value = sample.attributesList[node.attributeIndex]

        if value <= node.threshold:
            return self.recursivePredict(
                node.left,
                sample
            )
        else:
            return self.recursivePredict(
                node.right,
                sample
            )

    def predict(self, sample: Sample) -> int:
        return self.recursivePredict(self.root, sample)