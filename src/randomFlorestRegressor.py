from decisionTreeRegressor import DecisionTreeRegressor
from constants import NUM_TREES_REGRESSOR
from sample import Sample

from collections import Counter
import random
import math

class RandomFlorestRegressor:
    def __init__(self):
        self.trees = [DecisionTreeRegressor() for _ in range(NUM_TREES_REGRESSOR)]

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
        error = 0

        for sample in testSamples:
            prediction = self.predict(sample)

            error += (sample.output - prediction) ** 2

        return error / len(testSamples)

    def predict(self, sample: Sample) -> float:
        return sum([tree.predict(sample) for tree in self.trees]) / len(self.trees)
        