from decisionTree import DecisionTree
from sample import Sample
from constants import MIN_VARIANCE, MIN_SAMPLES

class DecisionTreeRegressor(DecisionTree):
    def __init__(self):
        super().__init__()

    def variance(self, items: list[float | int]) -> float:
        average = sum(items) / len(items)

        n = 0
        for item in items:
            n += (item - average) * (item - average)

        return n / len(items)

    def hasToStop(self, samples: list[Sample]) -> bool:
        return (
            self.variance([sample.output for sample in samples]) < MIN_VARIANCE or
            len(samples) < MIN_SAMPLES
        )

    def majorityOutput(self, samples: list[Sample]) -> float | int:
        return sum([sample.output for sample in samples]) / len(samples)

    def calculateAuxMeasure(self, trainingSamples: list[Sample]) -> float | int:
        outputs = [sample.output for sample in trainingSamples]
        return self.variance(outputs)

    def calculateMeasure(self, leftOutputs: list, rightOutputs: list, aux) -> float | int:
        leftVariance = self.variance(leftOutputs)
        rightVariance = self.variance(rightOutputs)

        outputsSize = len(leftOutputs) + len(rightOutputs)

        weightedError = (
            (len(leftOutputs) / outputsSize) * leftVariance
            +
            (len(rightOutputs) / outputsSize) * rightVariance
        )

        # aux is parent variance
        gain = aux - weightedError

        return gain

    def testTree(self, testSamples: list[Sample]) -> float:
        error = 0

        for sample in testSamples:
            prediction = self.recursivePredict(self.root, sample)

            error += (sample.output - prediction) ** 2

        return error / len(testSamples)
    