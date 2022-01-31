# load and display an image with Matplotlib
from matplotlib import  image
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import pandas as pd
import random
import matplotlib.patches as patches
import os

# load image as pixel array and LOAD it FROM CURRENT DIRECTORY
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
GroupImage = os.path.join(THIS_FOLDER, 'groupGray.jpg')
templateImage = os.path.join(THIS_FOLDER, 'boothiGray.jpg')

#Group and boothi Images
groupImg = image.imread(GroupImage)
targetImg = image.imread(templateImage)

# Rows and Columns Of Target and Group Image
rowsTargetImg = targetImg.shape[0]
colsTargetImg = targetImg.shape[1]

rowsGroupImg = groupImg.shape[0]
colsGroupImg = groupImg.shape[1]

# Generate  Random Population
def populInitialize(row, column, size):
    # Make Grid
    grid = np.zeros((row, column))
    population = []
    while len(population) < size:
        x, y = randint(0, len(grid) - 1), randint(0, len(grid[0]) - 1)
        population.append((x, y))
    return population

# Find Fitness Value
def fitnessScore(groupImg, targetImg, rndPopulation):
    rndImgs = []
    corrVal = []
    # y (rndPopulation[i][1]) is row, x(append(rndPopulation[i][0]) is col
    # Slice Big Image upto small image
    for i in range(len(rndPopulation)):
        rndImgs.append(
            groupImg[rndPopulation[i][1]:rndPopulation[i][1]+targetImg.shape[0],
                     rndPopulation[i][0]:rndPopulation[i][0]+targetImg.shape[1]])
    # Finding correlation
    for i in range(len(rndPopulation)):
        if len(rndImgs[i][0]) == len(targetImg[0]) and len(rndImgs[i]) == len(targetImg):
            numerator = np.mean(
                (rndImgs[i] - rndImgs[i].mean()) * (targetImg - targetImg.mean()))
            denominator = rndImgs[i].std() * targetImg.std()
            if denominator == 0:
                corrVal.append(0)
            else:
                result = numerator / denominator
                corrVal.append(result)
        else:
            corrVal.append(-1)
    corrVal = [round(num, 2) for num in corrVal]
    return corrVal

def Selection(rndPop, fitnessVal):
    rankPop = list(zip(fitnessVal, rndPop))
    rankPop.sort(reverse=True)
    # print(zipped[0][1])
    return rankPop

def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

# Function to do the recombinations
def crossOver(rankPop):
    evolvePop = []
    x_binary = []
    y_binary = []
    xyBin = []
    # After CrossOver Of XY-Parents
    evol_XY_par1 = []
    evol_XY_par2 = []
    # XY parents Evolved Lists
    evolXY = []
    # Convert Evolved Parents into the Decimal Number
    evolX_Dec = []
    evolY_Dec = []

    # Preserved the best fittest:
    retBestFit=[]
    retBestFit.append(rankPop.pop(0))
    retBestFit.append(rankPop.pop(1))
    
    # Convert ranked Population into Binary
    for i in range(len(rankPop)):
        x_binary.append(np.binary_repr(rankPop[i][1][0], width=10))
        y_binary.append(np.binary_repr(rankPop[i][1][1], width=9))

    xy_binary = [a + b for a, b in zip(x_binary, y_binary)]
    # print((xy_binary))

    for i in range(len(xy_binary)):
        xyBin.append([int(x) for x in str(xy_binary[i])])

    # xY-Parents
    xyParents = [xyBin[i:i+2] for i in range(0, len(xyBin), 2)]
    # print(xyParents)

    # CrossOver Between X and Y Parents
    cross_point = random.randint(0, 18)

    for i in range(len(xyParents)):
        evol_XY_par1.append(
            [(xyParents[i][0][0:cross_point + 1] + xyParents[i][1][cross_point+1:22])])
        evol_XY_par2.append(
            [(xyParents[i][1][0:cross_point + 1] + xyParents[i][0][cross_point+1:22])])

    # Merge Two Evolved Lists into one xy parents Crossover
    # mergeEvolXY = [a + b for a, b in zip(evol_XY_par1, evol_XY_par2)]
    mergeEvolXY = evol_XY_par1+evol_XY_par2
    # print(len(mergeEvolXY))

    for i in range(len(mergeEvolXY)):
        evolXY.append(mergeEvolXY[i][0])

    binEvolXY = []
    for j in range(len(evolXY)):
        binEvolXY.append([evolXY[j][i:i + 10]
                         for i in range(0, len(evolXY[j]), 10)])

    # Convert Binary to Decimal
    for i in range(len(binEvolXY)):
        evolX_Dec.append(binatodeci(binEvolXY[i][0]))
        evolY_Dec.append(binatodeci(binEvolXY[i][1]))
    # Merge Two evolved X and Y chromosomes into one evolved Population
    evolvePop = list(zip(evolX_Dec, evolY_Dec))
    # print(len(evolvePop))
    # print(retBestFit[0][1])
    evolvePop.insert(0, retBestFit[0][1])
    evolvePop.insert(1, retBestFit[1][1])
    return evolvePop

# Function for mutation
def mutation(evolPop):
    x_binary = []
    y_binary = []
    xBin = []
    yBin = []
    # print(evolPop,'\n')
    # Convert Evolved Parents into the Decimal Number
    evolX_Dec = []
    evolY_Dec = []
    # print(evolPop)

    # Preserved the best fittest:
    preBestP=[]
    preBestP.append(evolPop.pop(0))

    # Convert evolved Population into Binary
    for i in range(len(evolPop)):
        x_binary.append(np.binary_repr(evolPop[i][0], width=10))
        y_binary.append(np.binary_repr(evolPop[i][1], width=9))

    for i in range(len(x_binary)):
        xBin.append([int(x) for x in str(x_binary[i])])
        yBin.append([int(x) for x in str(y_binary[i])])

    rndPoint=random.randint(0,8)
    x = rndPoint
    y = random.randint(0,5)
    # rndRange = 4
    for x in range(5):
        i = random.randint(0,len(xBin)-1)
        if xBin[i][x] == 1:
            xBin[i][x] = 0
        else:
            xBin[i][x] = 1

        if yBin[i][y] == 1:
            yBin[i][y] = 0
        else:
            yBin[i][y] = 1

    for i in range(len(xBin)):
        evolX_Dec.append(binatodeci(xBin[i]))
        evolY_Dec.append(binatodeci(yBin[i]))

    mutPop = list(zip(evolX_Dec, evolY_Dec))
    mutPop.insert(0, preBestP[0])
    # print(evolPop)
    # bestFit=fitnessScore(groupImg, targetImg, mutPop)
    # SelPop=Selection(mutPop, bestFit)
    # print(SelPop)

# Experimenting for improving the mutation
    # mutPop = []
    # x_binary = []
    # y_binary = []
    # xyBin = []
    # # yBin=[]
    # #  Convert mutate individual into the Decimal Number
    # evolX_Dec = []
    # evolY_Dec = []

    # # # Preserved the best fittest:
    # preBestF=[]
    # preBestF.append(evolPop.pop(0))
    
    # # Convert ranked Population into Binary
    # for i in range(len(evolPop)):
    #     x_binary.append(np.binary_repr(evolPop[i][1][0], width=10))
    #     y_binary.append(np.binary_repr(evolPop[i][1][1], width=9))

    # xy_binary = [a + b for a, b in zip(x_binary, y_binary)]
    # # print((xy_binary))

    # for i in range(len(xy_binary)):
    #     xyBin.append([int(x) for x in str(xy_binary[i])])
    # # print(len(xyBin))
    # # xY-Parents

    # # for i in range(len(x_binary)):
    # #     xBin.append([int(x) for x in str(x_binary[i])])
    # #     yBin.append([int(x) for x in str(y_binary[i])])

    # rndPoint=np.random.randint(0,10)
    # # print(rndPoint)
    # x = rndPoint
    # # y = rndPoint
    # # rndRange = 4
    # for i in range():
    #     if xyBin[i][x] == 1:
    #         xyBin[i][x] = 0
    #     else:
    #         xyBin[i][x] = 1

    #     # if yBin[i][y] == 1:
    #     #     yBin[i][y] = 0
    #     # else:
    #     #     yBin[i][y] = 1


    # binEvolXY = []
    # for j in range(len(xyBin)):
    #     binEvolXY.append([xyBin[j][i:i + 10]
    #                      for i in range(0, len(xyBin[j]), 10)])
    
    # # print(len(binEvolXY))

    # # Convert Binary to Decimal
    # for i in range(len(binEvolXY)):
    #     evolX_Dec.append(binatodeci(binEvolXY[i][0]))
    #     evolY_Dec.append(binatodeci(binEvolXY[i][1]))


    # mutPop = list(zip(evolX_Dec, evolY_Dec))
    # bestFit=max(fitnessScore(groupImg, targetImg, mutPop))
    # mutPop.insert(0, preBestF[0][1])

    # bestFit1=max(fitnessScore(groupImg, targetImg, mutPop))
    # # SelPop=Selection(mutPop, bestFit)
    # print(bestFit,bestFit1,'\n')

    return mutPop

# Main
def EvolutionaryAlgo(generation, PopSize ,threshold):
    Population = populInitialize(
        groupImg.shape[1]-targetImg.shape[1], groupImg.shape[0]-targetImg.shape[0], PopSize)

    fitnessVal = fitnessScore(groupImg, targetImg, Population)

    rankedPop = Selection(Population, fitnessVal)

    PopulationEvolved=[rankedPop]
    keepPopEvolved=[]
    keepGenFitValues=[]
    
    for i in range(generation):
        evolvePop = crossOver(PopulationEvolved[0])
        keepPopEvolved.append(PopulationEvolved.pop(0))
        
        evlPopFitVal = fitnessScore(groupImg, targetImg, evolvePop)
        SelevlPopFitVal = Selection(evolvePop, evlPopFitVal)
        # PopulationEvolved.append(SelevlPopFitVal)

        mutatePop = mutation(evolvePop)

        mutPopFit = fitnessScore(groupImg, targetImg, mutatePop)
        keepGenFitValues.append(sorted(mutPopFit,reverse=True))

        SelMutPopFit = Selection(mutatePop, mutPopFit)

        PopulationEvolved.append(SelMutPopFit)

        if SelMutPopFit[0][0] >= threshold:
                
                # Represent Data using pandas Dataframe
                pd.set_option('display.max_rows', None)
                pndDt = pd.DataFrame({'Random-Population': Population,
                                      'Fitness-Values': fitnessVal,
                                    #   'Ranked-Pop': rankedPop,
                                    #   'Evolved-Pop': evolvePop,
                                      'Evolved-Pop-Fitness-Val': evlPopFitVal,
                                      'Ranked-Evolve-Pop': SelevlPopFitVal,
                                      'Mutate-Fitness-Val': mutPopFit,
                                      'Ranked-MutatePop':SelMutPopFit
                                      })
                print(pndDt,'\n')

                print('Target Image Found Successfully')
                print("Generation Populated Are: ", i)
                print('The Coordinates Of Target Image: ', SelMutPopFit[0][1])
                print('The Correlation Of Target Image: ',SelMutPopFit[0][0])


                # Show Target Image On The Group Image
                # fig, xy = plt.subplots()
                fig, (xy, ax2) = plt.subplots(1,2, figsize=(20,8))
                xy.imshow(groupImg, cmap='gray')
                Imgrect = patches.Rectangle(
                    (SelMutPopFit[0][1][0], SelMutPopFit[0][1][1]), 30, 30, linewidth=2, edgecolor='b', facecolor='none')
                xy.add_patch(Imgrect)
                # plt.show()

                # Finding max
                maxFitVal=[]
                meanFitVal=[]
                for i in range(len(keepGenFitValues)):
                    maxFitVal.append(keepGenFitValues[i][0])
                    # Finding Average
                    avg = np.mean(keepGenFitValues[i])
                    meanFitVal.append(round(avg,2))

                # maxFitVal.sort(reverse=False)
                # meanFitVal.sort(reverse=False)

                # Plots
                generations=range(1,len(maxFitVal)+1)
                plt.plot(generations,meanFitVal,maxFitVal)
                plt.title("Fittest Individual")
                plt.xlabel('Generations')
                plt.ylabel('Fitness Values')
                plt.show()

                return
        
        # return
        
    # print(keepPopEvolved,'\n')
    print('TARGET IMAGE IS NOT FOUND, SORRY')

#--------------------------------------------------------------------------------------------------------------------------------#


# Give generation, Population Size and threshold value:

PopulationSize=100
PopulationGeneration=5000
ThresholdValue=0.84

EvolutionaryAlgo(PopulationGeneration, PopulationSize, ThresholdValue)

#--------------------------------------------------------------------------------------------------------------------------------#