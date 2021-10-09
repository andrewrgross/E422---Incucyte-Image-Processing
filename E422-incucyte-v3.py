# -*- coding: utf-8 -*-
"""
E422 - Incucyte Image Data Extraction
Andrew R Gross, 2021-07-20

This program is intended to read in images, identify colonies, and extract key feature positions for further analysis.
"""

##############################################################################
### 1. Import Libraries
%reset -f

import os
import sys
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
from progress.bar import Bar

"""
from skimage.color import rgb2hsv
#from skimage import data, color
"""

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=3, infstr='inf',linewidth=200, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)

##############################################################################
### 2. Define functions

###### 2.1 - getImageFiles
def getImageFiles(targetDir = os.getcwd(), imgFormat = '.jpg'):
    #print(os.getcwd())
    allFiles = os.listdir(targetDir)
    imageFiles = []
    for file in allFiles:
        if imgFormat in file:
            imageFiles.append(file)
        else:
            pass
    print(str(len(imageFiles))+ ' files loaded from ' + str(targetDir))
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
    #for dateTime in dateTimeArray:    print(datetime.fromtimestamp(dateTime.item() / 10**9).strftime('%d_%b_%Y_%H:%M'))
    print('Start: ' + datetime.fromtimestamp(dateTimeArray[0].item() / 10**9).strftime('%d_%b_%Y_%H:%M'))
    print('End:   ' + datetime.fromtimestamp(dateTimeArray[-1].item() / 10**9).strftime('%d_%b_%Y_%H:%M'))

    wellNumberArray = imageMetadata['wellNumber'].unique()
    print('Unique wells found: ' + str(len(wellNumberArray)))
    for well in wellNumberArray:    print(well)

###### 2.4 - subsetDataset
def subsetDataset(imageMetadata, wellNumberN = 0):
    wellNumberArray = imageMetadata['wellNumber'].unique()
    print('Unique wells found: ' + str(len(wellNumberArray)))
    print('Well number ' + str(wellNumberN) + ' ("' + wellNumberArray[wellNumberN] + '") selected')
    imageMetadataCurrent = imageMetadata[imageMetadata['wellNumber'] == wellNumberArray[wellNumberN]]
    #imageMetadataCurrent = imageMetadataCurrent.sort_values('posNumber')
    return(imageMetadataCurrent)

###### 2.5 - nameDataset
def nameDataset(imageMetadata, selectedVessel, wellNumberN = 0):
    savetime = datetime.now().strftime('%d-%b-%y-%H%M')
    wellNumberArray = imageMetadata['wellNumber'].unique()
    selectedWell = wellNumberArray[wellNumberN]
    print('Well number ' + str(wellNumberN) + ' ("' + selectedWell + '") selected')
    datasetName = str(selectedVessel) +'_well-' + selectedWell + '_' + savetime
    print('Dataset name: ' + datasetName)
    return(datasetName)

###### 2.6  - stitchingImgFiles
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

###### 2.7  - memSize
def memSize(variable):
    return(str(round(getsizeof(variable)/1000000, 3)) + ' MB')

###### 2.8  - coloniesToContrours
def coloniesToContours(img, tolerance = 10, dataScaleFactor = 1):
    contours = find_contours(img, 0)
    sublistElements = countSublistElements(contours)
    print('Contour points: ' + str(sublistElements))

    coords_all = np.zeros((0,3)).astype(np.int32)

    counter = 1
    for contour in contours:
        coords = approximate_polygon(contour, tolerance= tolerance).astype(np.int32)
        newCol = np.ones((len(coords),1)).astype(np.int32())*counter                   #Measure the height and make a new column to add
        coords = np.concatenate((coords, newCol), axis = 1)         #Add the new column
        coords_all = np.concatenate((coords_all, coords), axis = 0) #Join the new coords into the old one
        counter += 1
        
  #  coords_all_x = coords_all[:,0].astype(np.int16)
  #  coords_all_y = coords_all[:,1].astype(np.int16)
  #  coords_all_c = coords_all[:,2].astype(np.int16)
    coordLength = len(coords_all)
    print('Reference points: ' + str(coordLength))
    ### Scale all coords by 1/10th to make them fit in a 16-bit value
    coords_all[:,0:2] = coords_all[:,0:2]/ dataScaleFactor
    
    ### Generate arrays of all xs and ys, then square, add, and sqrt to find hypotenuses
    coords_x_minus_x = np.ones((coordLength, coordLength)).astype(np.int32) * coords_all[:,0]
    #coords_x_minus_x = coords_x_minus_x * coords_all[:,0]
    coords_x_minus_x = coords_x_minus_x - np.transpose(coords_x_minus_x)
    coords_x_minus_x = np.square(coords_x_minus_x)
    
    coords_y_minus_y = np.ones((coordLength, coordLength)).astype(np.int32) * coords_all[:,1]
 #   coords_y_minus_y = coords_y_minus_y * coords_all[:,1]
    coords_y_minus_y = coords_y_minus_y - np.transpose(coords_y_minus_y)
    coords_y_minus_y = np.square(coords_y_minus_y)
    
    coords_dist = coords_x_minus_x + coords_y_minus_y
    coords_dist = np.sqrt(coords_dist, dtype = np.float32)*dataScaleFactor
    
    ###### 5.5 - Create a matrix of points from shared colonies  ################
    colony_matrix = np.ones((coordLength, coordLength)).astype(np.int32) * coords_all[:,2]
    colony_matrix = colony_matrix - np.transpose(colony_matrix)
    colony_matrix = ((colony_matrix == 0)*3000).astype(np.float32)
    
    coords_dist = coords_dist + colony_matrix
    del(colony_matrix, coords_x_minus_x, coords_y_minus_y)
    gc.collect()
    return([coords_dist, coords_all])

###### 2.9  - saveDistancesPlot
def saveDistancesPlot(img, closest_pairs, dataOutputDir, currentImageFile):
    plt.imshow(img)
    for row in closest_pairs:
        y_values = [row[0], row[2]]
        x_values = [row[1], row[3]]
        plt.plot(x_values, y_values, color = 'red', linewidth = 0.5)
    plt.text(50,3070, currentImgFile, color = 'White', fontsize = 11)
    plt.savefig(fname = dataOutputDir + '/' +currentImageFile+'-LINES.png', format='png')
    print(currentImageFile + 'saved in "' + dataOutputDir + '"')
    plt.show()

###### 2.10 - analyzeMemory
def analyzeMemory(allVar, returnTable = False):
    allVar
    memTable = pd.DataFrame(columns = ['Variable','Size'])
    for var in allVar:
        mem = getsizeof(eval(var))
        memTable = memTable.append({'Variable':var,'Size':mem}, ignore_index = True)
    memTable = memTable.sort_values(by = 'Size', ascending = False)
    print('Total memory usage is ' + str(round(memTable['Size'].sum()/1000000,3)) + 'MB')
    if(returnTable == True):
        return(memTable)
    else:
        pass

###### 2.11  - saveDistancesPlot
def countSublistElements(listOfLists):
    count = 0
    for element in listOfLists:
        count += len(element)
    return(count)

###### 2.12  - createDirIfNeeded
def createDirIfNeeded(fullpath, chdir = False):
    if os.path.exists(fullpath):
        if chdir:
            os.chdir(fullpath)
            print(fullpath + ' Exists. Assigned as output directory.')
        else:
            print(fullpath + ' Exists.')
    else:
        os.makedirs(fullpath)
        if chdir:
            os.chdir(fullpath)
            print(fullpath + ' Did not exist. Created and assigned as output directory.')
        else:
            print(fullpath + ' Did not exist. Created.')

##############################################################################
### 3. Import Images
###### 3.1 - Load metadata    #############################################
metadata = pd.read_csv('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\incucyte-metadata.csv')
print(metadata[['Vessel-ID','Vessel-name','Last-scan','Experimenter','Cell-type']])


###### 3.2 - Select set to analyze and import file list   #################
metadata['Analysis-ID']

selectedAnalysis = 112

targetDir = metadata['Directory'].tolist()[metadata['Analysis-ID'].tolist().index(selectedAnalysis)]
imageFiles = getImageFiles(targetDir)
os.chdir(targetDir)

###### 3.3 - Assign output directory   #################
variantSuffix = ''
dataOutputDir = 'E:/Incucyte/Analysis-output/A' + str(selectedAnalysis) + str(variantSuffix) + '/'
createDirIfNeeded(dataOutputDir)

print('Output directory:  ' + dataOutputDir)

##############################################################################
### 4. Pre-processing
###### 4.1 - Parse image files   #############################################
########## 4.1.1 - Create metadata table
imageMetadata = createMetadataTable(imageFiles)
summarizeDataset(imageMetadata, prefixN = 0, datetimeN= 0, wellNumberN = 0)
dateTimeN = 0

###### 4.2 - Subset files by well
wellList = []
for wellNumberN in range(0,len(imageMetadata['wellNumber'].unique())):
    print(str(wellNumberN))
    wellList.append(subsetDataset(imageMetadata, wellNumberN = wellNumberN))

    
###### 4.3 - Define scaling factor to convert px to um   #####################
micronsPerPx = 35000/3100
saWell = 3.14159*(35/2)**2

##############################################################################
### 5. Processing
###### 5.1 - Loop through wells  ###################################################

for imageMetadataCurrent in wellList:
    os.chdir(targetDir)
    fullDf = pd.DataFrame(columns = ['DataPoint', 'Time', 'SA-25', 'SA-50', 'SA-75', 'Surface Area-total', 'Colony Num', 'Confluence', 'Dist-avg', 'Dist-5%', 'Dist-25%', 'Dist-50%', 'Dist-75%', 'Dist-95%', 'N-Pairs'])
    for rowNum in range(0,len(imageMetadataCurrent)):
        print('\n##############################################################################################################################')
        print('PROCESSING FILE NUMBER ' + str(rowNum) + ' OF ' + str(len(imageMetadataCurrent)) + '. ' + str(round(rowNum/len(imageMetadataCurrent)*100, 1)) + '% COMPLETE.')
        ### Load current image from active row in provided table and plot
        currentImgFile = imageMetadataCurrent['imageFiles'].tolist()[rowNum]
        currentDateTime = imageMetadataCurrent['dateTime'].tolist()[rowNum]
        img = mpimg.imread(currentImgFile)
        print('Current file: ' + currentImgFile + '  ;  Dimentions: ' + str(img.shape) + '; Original size (MB): ' + memSize(img))
        img = closing(img > 0.5, square(3))     # Perform erosion to split colonies
        img = clear_border(img)           # remove artifacts connected to image border
        label_img = label(img)            # label image regions
        img_label_overlay = label2rgb(label_img, image=img, bg_label=0)  # to make the background transparent, pass the value of `bg_label' and leave `bg_color` as `None` and `kind` as `overlay`
        plt.imshow(img_label_overlay)
        #img_data_l = regionprops(label_image = label_img, intensity_image = img)
        img_data = pd.DataFrame(regionprops_table(label_image = label_img, intensity_image = img, properties = ('label', 'area', 'centroid' , 'eccentricity', 'equivalent_diameter')))
        img_data = img_data[img_data['area']>3]
        ### Free memory by deleting label_img and img_label_overlay
        #analyzeMemory(dir(), returnTable=False)
        del(label_img)
        del(img_label_overlay) 
        analyzeMemory(dir())
        # List the number of regions
        colonyN = len(img_data)
        print(str(colonyN) + ' colonies found')
        #img_data = pd.DataFrame(img_data_df)
        saTotal = round(img_data['area'].sum()*micronsPerPx**2/1000000,2)
        conf = round(saTotal /saWell,3)
        coords_out = coloniesToContours(img, tolerance = 10)
        coords_dist = coords_out[0]; coords_all = coords_out[1]
        
        min_dist = coords_dist.min(axis = 1)             # Find the minimum distance from each point to its closest neigbhor by taking the minimum value for each row of the coord_dist matrix
        closest_point = np.argmin(coords_dist, axis = 0) # 
        del(coords_dist)
        closest_pairs = np.hstack((coords_all[:,0:2],coords_all[closest_point,0:2]))
        del(coords_all)
        saveDistancesPlot(img, closest_pairs, dataOutputDir, currentImgFile)

        min_dist = (min_dist * micronsPerPx).astype(np.float32)
        ### Report and save values
        sa025    = round(np.quantile(img_data['area'], 0.25),1)
        sa05     = round(np.quantile(img_data['area'], 0.5),1)
        sa075    = round(np.quantile(img_data['area'], 0.75),1)
        npairs  = len(closest_pairs)
        davg    = round(np.mean(min_dist), 1)
        d005    = round(np.quantile(min_dist, 0.05),1)
        d025    = round(np.quantile(min_dist, 0.25),1)
        d05     = round(np.quantile(min_dist, 0.5),1)
        d075    = round(np.quantile(min_dist, 0.75),1)
        d095    = round(np.quantile(min_dist, 0.95),1)
        fullDf = fullDf.append({'DataPoint':currentImgFile, 'Time':currentDateTime, 'SA-25':sa025, 'SA-50':sa05, 'SA-75':sa075, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-avg':davg, 'Dist-5%':d005, 'Dist-25%':d025, 'Dist-50%':d05, 'Dist-75%':d075, 'Dist-95%':d095, 'N-Pairs':npairs}, ignore_index = True)
        gc.collect()
    ### Output fullDf
    createDirIfNeeded(dataOutputDir, chdir=True)
    outputName = nameDataset(imageMetadataCurrent, selectedVessel, 0)
    fullDf.to_excel(outputName + '.xls', index=False)



#with Bar('PROCESSING: ', fill='#',  max = len(imageMetadata), suffix='%(percent).1f%% - %(eta)ds') as bar:
    for rowNum in range(0,len(imageMetadata)):
        #print('\n###############################################################\n###############################################################')
        print('PROCESSING FILE NUMBER ' + str(rowNum) + ' OF ' + str(len(imageMetadata)) + '. ' + str(round(rowNum/len(imageMetadata)*100, 1)) + '% COMPLETE.')
        print('##############################################################################################################################\n')
        ### Load current image from active row in provided table and plot
        currentImgFile = imageMetadata['imageFiles'][rowNum]
        print('Current file: ' + currentImgFile)
        img = mpimg.imread(currentImgFile)
        print('Current file: ' + currentImgFile + '  ;  Dimentions: ' + str(img.shape) + '; Original size (MB): ' + memSize(img))
        
        img = closing(img > 0.5, square(3))     # Perform erosion to split colonies
        img = clear_border(img)           # remove artifacts connected to image border
        label_img = label(img)            # label image regions
        img_label_overlay = label2rgb(label_img, image=img, bg_label=0)  # to make the background transparent, pass the value of `bg_label' and leave `bg_color` as `None` and `kind` as `overlay`
        plt.imshow(img_label_overlay)
        img_data_l = regionprops(label_image = label_img, intensity_image = img)
        img_data_df = regionprops_table(label_image = label_img, intensity_image = img, properties = ('label', 'centroid', 'convex_image', 'area', 'eccentricity', 'equivalent_diameter', 'euler_number'))
        ### Free memory by deleting label_img and img_label_overlay
        #analyzeMemory(dir(), returnTable=False)
        del(label_img)
        del(img_label_overlay) 
        analyzeMemory(dir())
        # List the number of regions
        colonyN = len(img_data_l)
        print(str(colonyN) + ' colonies found')
        img_data = pd.DataFrame(img_data_df)
        saTotal = 10
        conf = 50
        coords_out = coloniesToContours(img, tolerance = 10)
        coords_dist = coords_out[0]; coords_all = coords_out[1]
        
        min_dist = coords_dist.min(axis = 1)             # Find the minimum distance from each point to its closest neigbhor by taking the minimum value for each row of the coord_dist matrix
        closest_point = np.argmin(coords_dist, axis = 0) # 
        del(coords_dist)
        closest_pairs = np.hstack((coords_all[:,0:2],coords_all[closest_point,0:2]))
        del(coords_all)
        #saveDistancesPlot(img, closest_pairs, dataOutputDir, currentImgFile)
        
        min_dist = (min_dist * img.shape[0]/35000).astype(np.float32)
        ### Report and save quantiles
        davg    = round(np.mean(min_dist, 1))
        d005    = round(np.quantile(min_dist, 0.05),1)
        d025    = round(np.quantile(min_dist, 0.25),1)
        d05     = round(np.quantile(min_dist, 0.5),1)
        d075    = round(np.quantile(min_dist, 0.75),1)
        d095    = round(np.quantile(min_dist, 0.95),1)

        #dMedian = round(np.median(min_dist),1)
        #dMin    = round(np.min(min_dist),1)
        #dSD1    = round(np.median(min_dist)-np.std(min_dist)) 
        #dSD2    = round(np.median(min_dist)+np.std(min_dist))
        #dMax    = round(np.max(min_dist),1)
        #print('For file ' + str(currentImgFile) + ' : ')
        #print('The median distance between colonies is ' + str(dMedian) + ' um')
        #print('The minimum distance between colonies is ' + str(dMin) + ' um')
        #print('The maximum distance between colonies is ' + str(dMax) + ' um')
        #print('Two-thirds of colonies are between ' + str(dSD1) + ' um and ' + str(dSD2) + ' um from neighboring colonies.')
        #fullDf = fullDf.append({'DataPoint':currentImgFile, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-min':dMin, 'Dist-SD-down':dSD1, 'Dist-med':dMedian, 'Dist-SD-up':dSD2, 'Dist-max':dMax}, ignore_index = True)
        fullDf = fullDf.append({'DataPoint':currentImgFile, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-avg':davg, 'Dist-5%':d005, 'Dist-25%':d025, 'Dist-50%':d05, 'Dist-75%':d075, 'Dist-95%':d095}, ignore_index = True)
        gc.collect()

        #bar.next()


createDirIfNeeded(dataOutputDir)
#io.imsave('VID64--11_Jul_21_1002--well_A1test.jpg', img, plugin=None, check_contrast=True)
outputName = nameDataset(imageMetadata, selectedVessel, wellNumberN)
fullDf.to_excel('incucyte-data-extraction-test.xls', index=False)


####################################################################
####################################################################

imageMetadataCurrent = subsetDataset(imageMetadata, prefixN = 0, datetimeN= dateNum, wellNumberN = 0)
currentImageFile = nameDataset(imageMetadata, prefixN = 0, datetimeN= dateNum, wellNumberN = 0)

imgStitched = mpimg.imread(imageMetadataCurrent['imageFiles'].iloc[0])
imgStitched.shape
imgStitched = rescale(imgStitched, 1/4, anti_aliasing = False)
###### 4.4 - Stitch well images  #############################################
#imgStitched = stitchImgFiles(imageMetadataCurrent, reductionFactor=4)


#plt.imshow(imgStitched)
memSize(imgStitched)

#imgStitched.dtype
#print(str(round(getsizeof(imgStitched)/1000000, 1)) + ' MB')

gc.collect()

##############################################################################
### 5. Processing
###### 5.1 - Isolate mask  ###################################################
#os.chdir('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\')

img = imgStitched
plt.imshow(img)
#io.imsave(str(currentImageFile + '-STICHED.jpg'), img, plugin=None, check_contrast=True)

img = closing(img > 0.5, square(3))     # Perform erosion to split colonies
img = clear_border(img)           # remove artifacts connected to image border
label_img = label(img)            # label image regions
img_label_overlay = label2rgb(label_img, image=img, bg_label=0)  # to make the background transparent, pass the value of `bg_label' and leave `bg_color` as `None` and `kind` as `overlay`
io.imshow(img_label_overlay)
#io.imsave(str(currentImageFile + '-HIGHLIGHTED.jpg'), img_label_overlay, plugin=None, check_contrast=True)

###### 5.2 - Define colony list  #############################################
# Compile the properties of each region
img_data_l = regionprops(label_image = label_img, intensity_image = img)
img_data_df = regionprops_table(label_image = label_img, intensity_image = img, properties = ('label', 'centroid', 'convex_image', 'area', 'eccentricity', 'equivalent_diameter', 'euler_number'))

# List the number of regions
colonyN = len(img_data_l)
print(colonyN)
img_data = pd.DataFrame(img_data_df)

saTotal = 10
conf = 50

###### 5.3 - Convert colonies to polygons ####################################


    
coords_dist = coloniesToContours(img)
min_dist = coords_dist.min(axis = 1)
closest_point = np.argmin(coords_dist, axis = 0)
closest_pairs = np.hstack((coords_all[:,0:2],coords_all[closest_point,0:2]))

plt.imshow(img)
for row in closest_pairs:
    y_values = [row[0], row[2]]
    x_values = [row[1], row[3]]
    plt.plot(x_values, y_values, color = 'red', linewidth = 1)
plt.show()

plt.imshow(img)
for row in contour:
    y_values = [row[0], row[1]]
    plt.plot(y_values, color = 'red', linewidth = 1)
    
plt.imshow(img)
plt.plot(contour[:,0], contour[:,1])
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

###################################
######################################################################
###################################

contours = find_contours(img, 0)

coords_all = np.zeros((1,3))

counter = 1
for contour in contours:
    #coords = approximate_polygon(contour, tolerance=2.5)
    coords = approximate_polygon(contour, tolerance=5)
    newCol = np.ones((len(coords),1))*counter                   #Measure the height and make a new column to add
    coords = np.concatenate((coords, newCol), axis = 1)         #Add the new column
    coords_all = np.concatenate((coords_all, coords), axis = 0) #Join the new coords into the old one
    counter += 1

#plt.scatter(coords[:, 1], coords[:, 0], '-r', linewidth=1.5)

plt.imshow(img,origin='lower')
plt.scatter(coords_all[:,1], coords_all[:,0], s = 1, c=(coords_all[:,2]))
plt.show()

plt.imshow(img,origin='lower')
plt.scatter(coords_all[:,1], coords_all[:,0], s = 1, c='Red')
plt.show()

plt.imshow(img,origin='lower')
plt.plot(contour[:,1], contour[:,0])
plt.show()

plt.imshow(img,origin='lower')
for contour in contours:
    plt.plot(contour[:,1], contour[:,0])
plt.show()

lenList = []
for contour in contours: lenList.append(len(contour))
contour = contours[lenList.index(max(lenList))]

coords = approximate_polygon(contour, tolerance=10)
#plt.imshow(img,origin='lower')
#plt.plot(coords[:,1], coords[:,0], s = 1)
plt.scatter(contour[:,1], contour[:,0], s = 0.1, c='Grey')
plt.scatter(coords[:,1], coords[:,0], s = 1, c='Red')
print(str(len(coords)) + ' of ' + str(len(contour)) + ' ... ' + str(round(len(coords)/len(contour)*100, 1)) + '%')
#plt.show()


###### 5.4 - Calculate the distance between all points  ######################
#coords_all = coords_all[100:200,0:3]   # For testing purposes
coords_all_x = coords_all[:,0].astype(np.int16)
coords_all_y = coords_all[:,1].astype(np.int16)
coords_all_c = coords_all[:,2].astype(np.int16)

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

saveDistancesPlot(img, closest_pairs, dataoutputDir, currentImageFile)

#io.imsave(str(currentImageFile + '-STICHED.jpg'), img, plugin=None, check_contrast=True)


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
fullDf

#fullDf = fullDf.append({'Experiment':currentPrefix, 'DateTime': currentDateTime, 'Date':currentDate, 'Time':currentTime, 'Well':currentWell, 'Surface Area-total':saTotal, 'Colony Num':colonyN, 'Confluence':conf, 'Dist-min':dMin, 'Dist-SD-down':dSD1, 'Dist-med':dMedian, 'Dist-SD-up':dSD2, 'Dist-max':dMax}, ignore_index = True)
#dfObj = dfObj.append({'User_ID': 23, 'UserName': 'Riti', 'Action': 'Login'}, ignore_index=True)


##############################################################################
### 6. Export
###### 6.1 - Save image    ###################################################
os.chdir('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\')

io.imsave('VID64--11_Jul_21_1002--well_A1test.jpg', img, plugin=None, check_contrast=True)

###### 6.2 - Save export dataframe  ##########################################
#matrixPlacements = pd.read_csv('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\matrixPlacements.csv')

fullDf.to_excel('incucyte-data-extraction-test.xls', index=False)

plt.imshow(img)
for row in closest_pairs:
    y_values = [row[0], row[2]]
    x_values = [row[1], row[3]]
    plt.plot(x_values, y_values, color = 'red', linewidth = 0.5)
plt.show()

os.chdir('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\')

im = Image.fromarray(img)
im.save("filename.jpeg")

 %reset -f



##############################################################################
### X. Scratchwork
###### X.1 - Scale image    ##################################################

dataOutputDir
for rowNum in range(0,len(imageMetadata)):
    print('PROCESSING FILE NUMBER ' + str(rowNum) + ' OF ' + str(len(imageMetadata)) + '. ' + str(round(rowNum/len(imageMetadata)*100, 1)) + '% COMPLETE.')
    print('##############################################################################################################################\n')
    ### Load current image from active row in provided table and plot
    currentImgFile = imageMetadata['imageFiles'][rowNum]
    img = mpimg.imread(currentImgFile)
    print('Current file: ' + currentImgFile + '  ;  Original dimentions: ' + str(img.shape) + '; Original size (MB): ' + memSize(img))
    img = rescale(img, 1/reductionFactor, anti_aliasing = False)
    print('Updated dimentions: ' + str(img.shape) + '; Updated size (MB): ' + memSize(img))
    im = Image.fromarray(img)
    im.save(dataOutputDir + currentImageFile+'-reduced.png', format='png')
    plt.savefig(fname = dataOutputDir+currentImageFile+'-LINES.png', format='png')
    print(currentImageFile + 'saved in "' + dataOutputDir + '"')
    plt.show()

os.chdir('C:\\Users\\grossar\\Box\\Sareen Lab Shared\\Data\\Andrew\\E422 - Incucyte data harvesting\\')

im = Image.fromarray(img)
im.save("filename.jpeg")