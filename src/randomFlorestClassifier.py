from decisionTreeClassifier import DecisionTreeClassifier
from constants import NUM_TREES_CLASSIFIER
from sample import Sample

from collections import Counter
import random
import math

class RandomFlorestClassifier:
    def __init__(self):
        self.trees = [DecisionTreeClassifier() for _ in range(NUM_TREES_CLASSIFIER)]

    def bootstrap(self, dataset: list) -> list:
        newDataset = []

        for _ in range(len(dataset)):
            randomIndex = random.randint(0, len(dataset)-1)
            newDataset.append(dataset[randomIndex])

        return newDataset

    def buildFlorest(self, trainingSamples: list[Sample]):
        for tree in self.trees:
            numAttributes = int(math.sqrt(len(trainingSamples[0].attributesList)))
            tree.buildTree(self.bootstrap(trainingSamples), numAttributes)

    def testFlorest(self, testSamples: list[Sample]) -> float:
        correct = 0

        for sample in testSamples:
            if self.predict(sample) == sample.output:
                correct += 1

        return correct / len(testSamples)

    def predict(self, sample: Sample) -> int:
        outputs = [tree.predict(sample) for tree in self.trees]
        return Counter(outputs).most_common(1)[0][0]