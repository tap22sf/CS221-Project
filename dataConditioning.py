import os
import csv
import numpy as np
import cv2
from zipfile import ZipFile
from PIL import Image
from shutil import copyfile

import prepRetinaNet


#
# Update an image dictionary extracted from the trainingCSV, by looking in a set of directories,
# and updating if dictionary if found
#
class ImageDescriptor:
    def __init__(self, imageID, detected, source, path):
        self.detected = detected
        self.imageID = imageID
        self.bboxes = []
        self.normalizedBBoxes = []
        self.source = source
        self.path = path
        self.labels = []
        self.img = None

        # Pointer to the label dictionary
        self.classDict = None

    def __str__(self):
        s = '{} -> {}\nLabels:\n'.format(self.imageID, self.path)
        for l in self.labels:
            label = l[1]
            confidence = int(l[2])
            if label in self.classDict and confidence:
                s += '{} confidence {} -> {}\n'.format(label, confidence, self.classDict[label])
            
        return s

    # Scale a bounding box to its image
    def prep(self):
    
        # Load the referenced image
        if self.detected:

            try:
                with Image.open(self.path) as self.img:
                    xsize, ysize = self.img.size
            except IOError:
                self.detected = False
                xsize, ysize = 0,0
                print ("Error: File does not appear to exist : {}".format(self.path))
                return

            #self.img = cv2.cvtColor(cv2.imread(self.path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            #
            #xsize = self.img.shape[1]
            #ysize = self.img.shape[0]
            #print ('ImageX:{} ImageY:{}'.format(xsize, ysize))
            
            # Look at all the boxes
            for b in self.bboxes:
                #print (b)
             
                xy = (int(float(b[4]) * xsize), int(float(b[6]) * ysize))   # (XMin1 * xsize, YMin1 * ysize)
                width = int(float(b[5]) * xsize) - xy[0]        # XMax1 * xsize - XMin1 * xsize
                height = int(float(b[7]) * ysize) - xy[1]       # YMax1 * ysize - Ymin * ysize 
                self.normalizedBBoxes.append((xy, width, height))

        else:
            print ('Cant prep...No image found:\n{}'.format(self.imageID))
        
    def addImage(self, subplot):
        
        print('Plotting:\n{}'.format(self))

        if len(self.img.shape) == 2 or self.img.shape[2] == 1: # Is image grayscale?
            plt.imshow(np.resize(image, (self.img.shape[0], self.image.shape[1])), interpolation="bicubic", cmap="gray")
        else:
            plt.imshow(self.img, interpolation="bicubic")

        # Pick a color and keep it the same for all matching boxes
        colors = ['y','g','r','b']
        chosenColors = {}
        boxCNum = 0
        boxIndex = 0
        for box in self.normalizedBBoxes:
            
            # Grab the box label and confidence
            boxLabel = self.bboxes[boxIndex][2]
            boxConfidence = int(self.bboxes[boxIndex][3])
            boxIndex += 1
            
            # Default color
            c = 'm'
            if boxConfidence < 1:
                # Low confidence box color
                c = 'w'
            elif boxLabel in chosenColors:
                c = chosenColors[boxLabel]
            else:
                c = colors[boxCNum]
                chosenColors[boxLabel] = c
                boxCNum += 1

            rect = patches.Rectangle((box[0]),box[1],box[2],linewidth=2,edgecolor=c,facecolor='none')
            subplot.add_patch(rect)


def extractAllImages(imageList, zipFiles, srcDir, destDir, extract, quick, noCopy, noScan):
    # If we are extracting from a zip -
    # Make sure destination Dir exists
    if not os.path.exists(destDir):
        os.mkdir(destDir)

    if extract:
        extracted = 0
        for zFile in zipFiles:
            with ZipFile(zFile + '.zip', 'r') as zip:
                # printing all the contents of the zip file
                print('Xtracting from: {}'.format(zFile))
                for f in zip.namelist():

                    imageID = os.path.splitext(os.path.basename(f))[0]

                    # Extract if found in the dictionary
                    if imageID in imageList:
                        imageList[imageID] = ImageDescriptor(imageID, True, zFile, destDir)
                        zip.extract(f, destDir)
                        extracted += 1

                        if extracted % 10000 == 0:
                            print(extracted)

                        if quick and extracted == 100:
                            return

    # Just scan the source directory and copy to the dest
    else:
        count = 0
        dirdict = {}

        print (srcDir)

        if not noScan:
            for dirName, subdirList, fileList in os.walk(srcDir):
                print('Found directory: %s' % dirName)
                print("Detected {} images in source directory".format(len(fileList)))
                for fname in fileList:
                    imageID = os.path.splitext(os.path.basename(fname))[0]
                    src = os.path.join(srcDir, dirName, fname)
                    dirdict[imageID] = src


        # Extract if found in the dictionary
        for imageID in imageList:

            if imageID in dirdict or noScan:
                fn = imageID + '.jpg'

                # Copy to the destination if missing
                dst = os.path.join(destDir, fn)
                imageList[imageID] = ImageDescriptor(imageID, True, "dirscan", dst)

                if not noCopy:
                    if not os.path.isfile(dst):
                        src = dst
                        if not noScan:
                            src = dirdict[imageID]
                            
                            if not os.path.isfile(src):
                                print ("Copy - but file missing in Src : {}".format(src))
                            else:
                                copyfile(src, dst)

                else:
                    if not os.path.isfile(dst):
                        print ("No Copy - but file missing in dest : {}".format(dst))

                count += 1
                if count % 10000 == 0:
                    print(count)

                if quick and count == 100:
                    return

    return

def extractChallengeImages(datasetCSV, srcDir, destinationDirectory, extract, quick, noCopy, noScan):

    # First column is the image list
    trainingImageList = {}
    with open(os.path.join(datasetCSV)) as f:
        csvreader = csv.reader(f)
        next(csvreader)
        count = 0
        for row in csvreader:
            # An dictionary of "ID, found, location"
            trainingImageList[row[0]] = ImageDescriptor(row[0], False, "csv", "")
            count += 1

    print("Detected {} images in CSV".format(count))

    # List of directories with unzipped image files
    imageFiles = [          'G:\\train_00',
                            'G:\\train_01',
                            'G:\\train_02',
                            'G:\\train_03',
                            'G:\\train_04',
                            'D:\\train_05',
                            'D:\\train_06',
                            'D:\\train_07',
                            'D:\\train_08']

    extractAllImages(trainingImageList, imageFiles, srcDir, destinationDirectory, extract, quick, noCopy, noScan)
    
    total = 0
    used = 0

    newDict = {}
    for imageID, ID in trainingImageList.items():  
        total += 1
        if ID.detected:
            used += 1
            newDict[imageID] = ID

        else:
            if not quick: print ('Missing: {}'.format(ID.imageID))

    print ('Image references in training Set: {} Found in file search or zips: {}'.format(used, total))
    return newDict

def readBBoxes(imageList, BBoxCSV):

    # First column is the image list
    with open(os.path.join(BBoxCSV)) as f:
        csvreader = csv.reader(f)
        next(csvreader)
        matched = 0
        notMatched = 0
        unique = 0
        for row in csvreader:
            # An dictionary of "ID, found, location"
            imageID = row[0]
            if imageID in imageList:
                if imageList[imageID].bboxes == []:
                    unique += 1
                
                imageList[imageID].bboxes.append(row)
                matched += 1
            else:
                notMatched += 1
    
    print ('BBoxes detected in training Set: {} for {} images,  Found in CSV: {}'.format(matched, unique, notMatched+matched))

    for imageID, ID in imageList.items():
        if ID.bboxes == []:
            print('No Boxxes for {}'.format(ID))
                
    return

def readLabels(imageList,labelsCSV,classDict):

    # First column is the image list
    with open(os.path.join(labelsCSV)) as f:
        csvreader = csv.reader(f)
        next(csvreader)
        matched = 0
        notMatched = 0
        unique = 0
        for row in csvreader:
            # An dictionary of "ID, LabelName, Confidencen"
            imageID = row[0]
            if imageID in imageList:
                if imageList[imageID].labels == []:
                    unique += 1
                
                imageList[imageID].labels.append(row)
                imageList[imageID].classDict = classDict
                matched += 1
            else:
                notMatched += 1
    
    print ('Labels detected in training Set: {} in {} images,  Total in CSV: {}'.format(matched, unique, notMatched+matched))

    for imageID, ID in imageList.items():
        if ID.labels == []:
            print('No Labels for {}'.format(ID))
                

def readClasses(classesCSV):

    classDict = {}
    print ("Loading Class Info")

    # First column is the image list
    with open(os.path.join(classesCSV)) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            # An dictionary of "ID, String"
            classID = row[0]
            if classID not in classDict:
                classDict[classID] = row[1]
            else:
                print ('Duplicate Class ID: {} with {}, {}'.format(classID, classDict[classID], row[1]))
    
    return classDict    

def writeRetinanetTrainCSV(imageList, trainbCSVName, devCSVName, lcsvName, trainSplit):

    # Open up the new outputfile
    usedClassNames = {}
    index = 0
    processed = 0

    print('Generating RetinaNet training files')

    # Open up the new outputfiles
    with open(os.path.join(trainbCSVName), 'w', newline='') as ft:
        trainWriter = csv.writer(ft)

        with open(os.path.join(devCSVName), 'w', newline='') as fd:
            devWriter = csv.writer(fd)

            with open(os.path.join(lcsvName), 'w', newline='') as fl:
                labelWriter = csv.writer(fl)

                count = 0
                total = len(imageList)
                print ("Writing Training Files for {} images".format(total))
                for id, ID in imageList.items():
                    if ID.path:

                        ID.prep()
                        if ID.detected:
                            xsize, ysize = ID.img.size
                        else:
                            continue

                        count += 1

                        for bbox in ID.bboxes:
                            label = bbox[2]
                            l = ID.classDict[label]
                            x1 =  int(float(bbox[4]) *xsize)
                            y1 =  int(float(bbox[6]) *ysize)
                            x2 =  int(float(bbox[5]) *xsize)
                            y2 =  int(float(bbox[7]) *ysize)

                            # if x1 > x2:
                            #     x1, x2 = x2, x1
                            #     y1, y2 = y2, y1

                            if x2 <= x1 or y2 <= y1:
                                print ('Ignoring BBox on {} : {},{} {},{}'.format(id, x1, y1, x2, y2))
                                continue

                            # Split train dev
                            if count < total * trainSplit:
                                trainWriter.writerow([ID.path, x1, y1, x2, y2, l])
                            else:
                                devWriter.writerow([ID.path, x1, y1, x2, y2, l])

                            processed += 1
                            if processed % 10000 == 0:
                                print(processed)

                            if l not in usedClassNames:
                                usedClassNames[l] = index
                                labelWriter.writerow([l, index])
                                index += 1


                    else:
                        print ('No path for {}'.format(id))



# Input Raw Datafiles
trainImageDatsetCSV       = 'ChallengeMetaData\\challenge-2018-train-vrd.csv'
trainBBoxDatsetCSV        = 'ChallengeMetaData\\challenge-2018-train-vrd-bbox.csv'
trainLabelsDatsetCSV      = 'ChallengeMetaData\\challenge-2018-train-vrd-labels.csv'
trainClassesDatsetCSV     = 'ChallengeMetaData\\challenge-2018-classes-vrd.csv'
fullDatasetDir            = 'F:\TrainingImages'

retinaNetTrainCSV         = 'Output\\retinaNetTrain.csv'
retinaNetDevCSV           = 'Output\\retinaNetDev.csv'
retinaNetTestCSV          = 'Output\\retinaNetTest.csv'
retinaNetClassCSV         = 'Output\\retinaNetClass.csv'

desitnationDir            = 'F:\DataTest'

valImageDatsetCSV         = 'ChallengeMetaData\\challenge-2018-image-ids-valset-vrd.csv'
trainImages               = 'ChallengeMetaData\\train-images-boxable-with-rotation.csv'

# Updated to work with SaturnV
super = False
if super:
    trainImageDatsetCSV       = './ChallengeMetaData/challenge-2018-train-vrd.csv'
    trainBBoxDatsetCSV        = './ChallengeMetaData/challenge-2018-train-vrd-bbox.csv'
    trainLabelsDatsetCSV      = './ChallengeMetaData/challenge-2018-train-vrd-labels.csv'
    trainClassesDatsetCSV     = './ChallengeMetaData/challenge-2018-classes-vrd.csv'
    fullDatasetDir            = '/home/dataset/OpenImagesV4/train'

    retinaNetTrainCSV         = 'Output/retinaNetTrain.csv'
    retinaNetDevCSV           = 'Output/retinaNetDev.csv'
    retinaNetTestCSV          = 'Output/retinaNetTest.csv'
    retinaNetClassCSV         = 'Output/retinaNetClass.csv'

    desitnationDir            = '/workspace/TrainingImages'

    valImageDatsetCSV         = './ChallengeMetaData/challenge-2018-image-ids-valset-vrd.csv'
    trainImages               = './ChallengeMetaData/train-images-boxable-with-rotation.csv'


# Extract from Zip files?
extract = False
noCopy = False
noScan = True    # Just rebuild the training files from the dest dir

# Shorten the dataset?
quick = False

# Size of the training set
trainSplit = 0.8

print ("Starting Training Set Image Extraction")
detectedImages = extractChallengeImages(trainImageDatsetCSV, fullDatasetDir, desitnationDir, extract, quick, noCopy, noScan)

# Set up classes decoder
classDict = readClasses(trainClassesDatsetCSV)

# Load the label dataset and update the image database and attach the class dictionary
readLabels(detectedImages, trainLabelsDatsetCSV, classDict)

# Now check the bounding boxes against the images and update the image database
readBBoxes(detectedImages, trainBBoxDatsetCSV)

# Generate a csvfile for training that is retinanet friendly - also do the dev split
writeRetinanetTrainCSV(detectedImages, retinaNetTrainCSV, retinaNetDevCSV, retinaNetClassCSV, trainSplit)

