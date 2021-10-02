# -*- coding: utf-8 -*-
"""
E422 - Incucyte Image Data Extraction
Andrew R Gross, 2021-07-20

This program is intended to read in images, identify colonies, and extract key feature positions for further analysis.
"""

##############################################################################
### 1. Import Libraries

import os
from sys import getsizeof
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from datetime import datetime
from skimage import data, io, filters
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gc


"""
from skimage.color import rgb2hsv
#from skimage import data, color
"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=3, infstr='inf',linewidth=200, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)
#import scipy

### Import matix placements
matrixPlacements = pd.read_csv('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\matrixPlacements.csv')
##############################################################################
### 2. Define functions

###### 2.1 - getImageFiles
def getImageFiles(imgFormat = '.jpg'):
    print(os.getcwd())
    allFiles = os.listdir()
    imageFiles = []
    for file in allFiles:
        if imgFormat in file:
            print(file)
            imageFiles.append(file)
        else:
            pass
    return(imageFiles)

###### 2.2 - createMetadataTable 
def createMetadataTable (imageFiles):
    prefix     = []
    wellNumber = []
    posNumber  = []
    dateValue  = []
    timeValue  = []
    dateTime   = []
    for fileName in imageFiles:
        splitName = fileName.split('.')[0]
        splitName = splitName.split('_')
        prefix     += [splitName[0]]
        wellNumber += [splitName[1]]
        posNumber  += [int(splitName[2])]
        dateValue  += [splitName[3]]
        timeValue  += [splitName[4]]
        dateTime   += [datetime.strptime(splitName[3]+splitName[4], '%Yy%mm%dd%Hh%Mm')]
    imageMetadata = pd.DataFrame(list(zip(imageFiles, prefix, wellNumber, posNumber, dateValue, timeValue, dateTime)), columns = ['imageFiles', 'prefix', 'wellNumber', 'posNumber', 'dateValue', 'timeValue', 'dateTime'])
    return(imageMetadata)

###### 2.3 - summarizeDataset
def summarizeDataset(imageMetadata, prefixN = 0, datetimeN = 0, wellNumberN = 0):
    prefixArray = imageMetadata['prefix'].unique()
    print('Unique prefixes found: ' + str(len(prefixArray)))
    for prefix in prefixArray:    print(prefix)

    dateTimeArray = imageMetadata['dateTime'].unique()
    print('Unique date-times found: ' + str(len(dateTimeArray)))
    for dateTime in dateTimeArray:    print(datetime.fromtimestamp(dateTime.item() / 10**9).strftime('%d_%b_%y'))

    wellNumberArray = imageMetadata['wellNumber'].unique()
    print('Unique wells found: ' + str(len(wellNumberArray)))
    for well in wellNumberArray:    print(well)

###### 2.4 - subsetDataset
def subsetDataset(imageMetadata, prefixN = 0, datetimeN = 0, wellNumberN = 0):
    prefixArray = imageMetadata['prefix'].unique()
    print('Unique prefixes found: ' + str(len(prefixArray)))
    for prefix in prefixArray:    print(prefix)
    print('Prefix number ' + str(prefixN) + ' ("' + prefixArray[prefixN] + '") selected')
    imageMetadataCurrent = imageMetadata[imageMetadata['prefix'] == prefixArray[prefixN]]

    dateTimeArray = imageMetadataCurrent['dateTime'].unique()
    print('Unique date-times found: ' + str(len(dateTimeArray)))
    for dateTime in dateTimeArray:    print(datetime.fromtimestamp(dateTime.item() / 10**9).strftime('%d_%b_%y'))
    currentDatetime = dateTimeArray[datetimeN]
    selectedDatetime = datetime.fromtimestamp(currentDatetime.item() / 10**9).strftime('%d_%b_%y_%H:%M')
    print('DateTime number ' + str(datetimeN) + ' (' + selectedDatetime + ') selected')
    imageMetadataCurrent = imageMetadataCurrent[imageMetadataCurrent['dateTime'] == currentDatetime]

    wellNumberArray = imageMetadataCurrent['wellNumber'].unique()
    print('Unique wells found: ' + str(len(wellNumberArray)))
    for well in wellNumberArray:    print(well)
    print('Well number ' + str(wellNumberN) + ' ("' + wellNumberArray[wellNumberN] + '") selected')
    imageMetadataCurrent = imageMetadataCurrent[imageMetadataCurrent['wellNumber'] == wellNumberArray[wellNumberN]]
    
    imageMetadataCurrent = imageMetadataCurrent.sort_values('posNumber')

    return(imageMetadataCurrent)

###### 2.5 - nameDataset
def nameDataset(imageMetadata, prefixN = 0, datetimeN = 0, wellNumberN = 0):
    prefixArray = imageMetadata['prefix'].unique()
    selectedPrefix = prefixArray[prefixN]
    print('Prefix number ' + str(prefixN) + ' ("' + selectedPrefix + '") selected')
    imageMetadataCurrent = imageMetadata[imageMetadata['prefix'] == selectedPrefix]

    dateTimeArray = imageMetadataCurrent['dateTime'].unique()
    currentDatetime = dateTimeArray[datetimeN]
    selectedDatetime = datetime.fromtimestamp(currentDatetime.item() / 10**9).strftime('%d_%b_%y_%H:%M')
    print('DateTime number ' + str(datetimeN) + ' (' + selectedDatetime + ') selected')
    imageMetadataCurrent = imageMetadataCurrent[imageMetadataCurrent['dateTime'] == currentDatetime]

    wellNumberArray = imageMetadataCurrent['wellNumber'].unique()
    selectedWell = wellNumberArray[wellNumberN]
    print('Well number ' + str(wellNumberN) + ' ("' + selectedWell + '") selected')

    datasetName = str(selectedPrefix + '--' + selectedDatetime + '--well_' + selectedWell)
    print('Dataset name: ' + datasetName)
    return(datasetName)

###### 2.4  - stitchingImgFiles
def stitchImgFiles(imageMetadataCurrent, reductionFactor = 4):
    print('Stitching ' + str(len(imageMetadataCurrent)) + ' images into 7x9 grid:')
    ### Define empty matrix to assign images into
    imgFirst = mpimg.imread(imageMetadataCurrent['imageFiles'].iloc[0])
    yw = np.shape(imgFirst)[0]/reductionFactor
    xw = np.shape(imgFirst)[1]/reductionFactor
    imgStitched = np.zeros((int(yw*9), int(xw*7))).astype(np.uint8)
    ### Loop through each and assign
    for posNum in range(0,len(imageMetadataCurrent)):
        file = imageMetadataCurrent['imageFiles'].iloc[posNum]
        print(file)
        imgNew=mpimg.imread(file)
        imgNew = rescale(imgNew, 1/reductionFactor, anti_aliasing = False)
        yn = matrixPlacements['y'][posNum]
        xn = matrixPlacements['x'][posNum]
        imgStitched[int(yn*yw):int((yn+1)*yw), int(xn*xw):int((xn+1)*xw)] = imgNew
    return(imgStitched)

def memSize(variable):
    print(str(round(getsizeof(variable)/1000000, 1)) + ' MB')
    
##############################################################################
### 3. Import Images
###### 3.1 - Test load images    #############################################

os.chdir('E:\\Andrew\\Incucyte')
#os.chdir('C:\\Users\\grossar\\Box\\MTEC grant\\Incucyte\\Andrew')

imageFiles = getImageFiles()

###### 3.2 - Generate an empty df   ##########################################
#fullDf = pd.DataFrame(columns = ['Experiment', 'DateTime', 'Date', 'Time', 'Well', 'Surface Area-total', 'Colony Num', 'Confluence', 'Dist-min', 'Dist-SD-down', 'Dist-med', 'Dist-SD-up', 'Dist-max'])
fullDf = pd.DataFrame(columns = ['DataPoint', 'Surface Area-total', 'Colony Num', 'Confluence', 'Dist-min', 'Dist-SD-down', 'Dist-med', 'Dist-SD-up', 'Dist-max'])

##############################################################################
### 4. Pre-processing
###### 4.1 - Parse image files   #############################################
########## 4.1.1 - Create metadata table
imageMetadata = createMetadataTable(imageFiles)

########## 4.1.2 - Subset metadata by project, date, and time
### Report prefix groups present:

summarizeDataset(imageMetadata, prefixN = 0, datetimeN= 0, wellNumberN = 0)

imageMetadataCurrent = subsetDataset(imageMetadata, prefixN = 0, datetimeN= 0, wellNumberN = 0)

currentImageFile = nameDataset(imageMetadata, prefixN = 0, datetimeN= 0, wellNumberN = 0)

###### 4.2 - Stitch well images  #############################################
imgStitched = stitchImgFiles(imageMetadataCurrent, reductionFactor=4)
io.imshow(imgStitched)
memSize(imgStitched)

#imgStitched.dtype
#print(str(round(getsizeof(imgStitched)/1000000, 1)) + ' MB')

gc.collect()

##############################################################################
### 5. Processing
###### 5.1 - Isolate mask  ###################################################

#img = rgb2hsv(img)[:, :, 1]   # Extract the saturation layer

img = imgStitched
img = closing(img > 0.5, square(9))

# remove artifacts connected to image border
s_cleared = clear_border(img)

# label image regions
label_img = label(s_cleared)
#label_img = label(img)

# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`

s_img = img
img_label_overlay = label2rgb(label_img, image=s_img, bg_label=0)
io.imshow(img_label_overlay)

# Compile the properties of each region
img_data_l = regionprops(label_image = label_img, intensity_image = img)
img_data_df = regionprops_table(label_image = label_img, intensity_image = img, properties = ('label', 'centroid', 'convex_image', 'area', 'eccentricity', 'equivalent_diameter', 'euler_number'))


saTotal = 10
conf = 50


# List the number of regions
colonyN = len(img_data_l)
print(colonyN)
img_data = pd.DataFrame(img_data_df)


###### 5.2 - Define colony list  #############################################


###### 5.3 - Convert colonies to polygons ####################################

contours = find_contours(img, 0)
# approximate / simplify coordinates of the two ellipses
coords_all = np.array([[0],[0],[0]])
coords_all = np.zeros((1,3))

counter = 1
for contour in contours:
    #coords = approximate_polygon(contour, tolerance=2.5)
    coords = approximate_polygon(contour, tolerance=5)
    #Measure the height and make a new column to add
    newCol = np.ones((len(coords),1))*counter
    #Add the new column
    coords = np.concatenate((coords, newCol), axis = 1)
    #Join the new coords into the old one
    coords_all = np.concatenate((coords_all, coords), axis = 0)
    counter += 1

#plt.scatter(coords[:, 1], coords[:, 0], '-r', linewidth=1.5)

plt.imshow(img)
plt.scatter(coords_all[:,1], coords_all[:,0], s = 3, c=(coords_all[:,2]))
plt.show()
###### 5.4 - Calculate the distance between all points  ######################
#coords_all = coords_all[100:200,0:3]   # For testing purposes

coords_all_x = coords_all[:,0].astype(np.int16)
coords_all_y = coords_all[:,1].astype(np.int16)
coords_all_c = coords_all[:,2].astype(np.int16)

### For testing & dev, make them small
#coords_all_x = coords_all_x[200:220]
#coords_all_y = coords_all_y[200:220]
#coords_all_c = coords_all_c[200:220]

coords_x_minus_x = np.ones((len(coords_all_x),len(coords_all_x)))
coords_x_minus_x = coords_x_minus_x * coords_all_x
coords_x_minus_x = coords_x_minus_x - np.transpose(coords_x_minus_x)
coords_x_minus_x = np.square(coords_x_minus_x)

coords_y_minus_y = np.ones((len(coords_all_y),len(coords_all_y)))
coords_y_minus_y = coords_y_minus_y * coords_all_y
coords_y_minus_y = coords_y_minus_y - np.transpose(coords_y_minus_y)
coords_y_minus_y = np.square(coords_y_minus_y)

coords_dist = coords_x_minus_x + coords_y_minus_y
coords_dist = np.sqrt(coords_dist)

###### 5.5 - Create a matrix of points from shared colonies  ################
colony_matrix = np.ones((len(coords_all_c), len(coords_all_c))) * coords_all_c
colony_matrix = colony_matrix - np.transpose(colony_matrix)
colony_matrix = (colony_matrix == 0)*1
#colony_matrix = (colony_matrix == 0) * np.Inf
#colony_matrix == 0 = 10000

###### 5.6 - Measure distances between key points  ###########################
# 
### To take the mininum for each row, I need to remove the 0s.
coords_dist = coords_dist + colony_matrix*100000
min_dist = coords_dist.min(axis = 1)


###### 5.7 - PLot distances between points ###################################

########## 5.7.1 - Find the index of the minimum value for each point 
# closest_point = np.where(coords_dist == min_dist)[0]
closest_point = np.argmin(coords_dist, axis = 0)

closest_pairs = np.hstack((coords_all[:,0:2],coords_all[closest_point,0:2]))


plt.imshow(img)
for row in closest_pairs:
    y_values = [row[0], row[2]]
    x_values = [row[1], row[3]]
    plt.plot(x_values, y_values, color = 'red', linewidth = 1)
plt.show()

#min_dist = min_dist / 0.295
min_dist = min_dist * 2.82

dMedian = round(np.median(min_dist),1)
dMin    = round(np.min(min_dist),1)
dSD1    = round(np.median(min_dist)-np.std(min_dist)) 
dSD2    = round(np.median(min_dist)+np.std(min_dist))
dMax    = round(np.max(min_dist),1)

print('For file ' + str(currentImageFile) + ' : ')
print('The median distance between colonies is ' + str(dMedian) + ' um')
print('The minimum distance between colonies is ' + str(dMin) + ' um')
print('The maximum distance between colonies is ' + str(dMax) + ' um')
print('Two-thirds of colonies are between ' + str(dSD1) + ' um and ' + str(dSD2) + ' um from neighboring colonies.')

###### 5.8 - Add new stats to a row        ###################################

fullDf = fullDf.append({'DataPoint':currentImageFile, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-min':dMin, 'Dist-SD-down':dSD1, 'Dist-med':dMedian, 'Dist-SD-up':dSD2, 'Dist-max':dMax}, ignore_index = True)

#fullDf = fullDf.append({'Experiment':currentPrefix, 'DateTime': currentDateTime, 'Date':currentDate, 'Time':currentTime, 'Well':currentWell, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-min':dMin, 'Dist-SD-down':dSD1, 'Dist-med':dMedian, 'Dist-SD-up':dSD2, 'Dist-max':dMax}, ignore_index = True)
#dfObj = dfObj.append({'User_ID': 23, 'UserName': 'Riti', 'Action': 'Login'}, ignore_index=True)


##############################################################################
### 6. Export
###### 6.1 - Save image    ###################################################
os.chdir('E:\\Andrew\\Incucyte')
os.chdir('C:\\Users\\grossar\\Box\\MTEC grant\\Incucyte\\Andrew')
###### 6.2 - Save export dataframe  ##########################################
#matrixPlacements = pd.read_csv('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\matrixPlacements.csv')
os.chdir('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\')

fullDf.to_excel('incucyte-data-extraction-test.xls', index=False)

im = Image.fromarray(array)
im.save("filename.jpeg")
