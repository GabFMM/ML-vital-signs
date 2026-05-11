from constants import *
from sample import Sample

from typing import Literal
import random

def buildSample(line: str, modelType: Literal["classifier", "regressor"]) -> Sample:
    lineList = line.strip().split(",")

    id = int(lineList[0])

    # ignores: id (0), inrelevant data (1 and 2) and outputs (6 and 7)
    attributesStr = lineList[3:6]

    attributesFloat = [float(attr) for attr in attributesStr]

    if modelType == "regressor":
        output = float(lineList[6])
    elif modelType == "classifier":
        output = int(lineList[7])

    return Sample(id, attributesFloat, output)

def trainingSet(modelType: Literal["classifier", "regressor"]) -> tuple[list[Sample], set[int]]:
    with open(DATA_FILE_PATH, "r") as f:
        lines = f.readlines()

    numLines = len(lines)

    trainingSetSize = int(numLines * TRAINING_SET_PERCENT)

    chosenIndexes = set(
        random.sample(range(numLines), trainingSetSize)
    )

    samples = []
    for i, line in enumerate(lines):
        if i in chosenIndexes:
            samples.append(buildSample(line, modelType))

    return samples, chosenIndexes

def testSet(choosenIndexes: set[int], modelType: Literal["classifier", "regressor"]) -> list[Sample]:
    with open(DATA_FILE_PATH, "r") as f:
        samples = []
        for i, line in enumerate(f):
            if i not in choosenIndexes:
                samples.append(buildSample(line, modelType))

    return samples
                
if __name__ == "__main__":
    