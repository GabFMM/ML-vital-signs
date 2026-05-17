from decisionTree import DecisionTree
from node import Node

from sample import Sample
from math import log2
from collections import Counter
import random

class DecisionTreeClassifier(DecisionTree):
    def __init__(self):
        super().__init__()

    def majorityOutput(self, samples: list[Sample]) -> float | int:
        outputs = [
            sample.output
            for sample in samples
        ]

        counts = Counter(outputs)

        return counts.most_common(1)[0][0]

    def hasToStop(self, samples: list[Sample]) -> bool:
        firstOutput = samples[0].output

        for sample in samples:
            if sample.output != firstOutput:
                return False

        return True

    def entropy(self, values: list) -> float:
        counts = Counter(values)

        entropySum = 0

        for occurrences in counts.values():

            p = occurrences / len(values)

            entropySum += p * log2(p)

        return -entropySum

    def calculateAuxMeasure(self, trainingSamples: list[Sample]) -> float | int:
        outputs = [
            sample.output
            for sample in trainingSamples
        ]
        return self.entropy(outputs) # parent entropy

    def calculateMeasure(self, leftOutputs: list, rightOutputs: list, aux) -> float | int:
        leftEntropy = self.entropy(leftOutputs)
        rightEntropy = self.entropy(rightOutputs)

        trainingSamplesSize = len(leftOutputs) + len(rightOutputs)

        # weighted entropy
        splitEntropy = (
            (len(leftOutputs) / trainingSamplesSize) * leftEntropy
            + (len(rightOutputs) / trainingSamplesSize) * rightEntropy
        )

        # aux is parent entropy
        # return information gain
        return aux - splitEntropy
    
    # return the average number of correct predicts
    def testTree(self, testSamples: list[Sample]) -> float:
        correct = 0

        for sample in testSamples:
            if sample.output == self.recursivePredict(self.root, sample):
                correct += 1

        return correct / len(testSamples)