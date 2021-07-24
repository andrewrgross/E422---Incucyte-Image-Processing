# -*- coding: utf-8 -*-
"""
E422 - Incucyte Image Data Extraction
Andrew R Gross, 2021-07-20

This program is intended to read in images, identify colonies, and extract key feature positions for further analysis.
"""

##############################################################################
### 1. Import Libraries

import os
from skimage import data, io, filters
from skimage.color import rgb2hsv
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=3, infstr='inf',linewidth=200, nanstr='nan', precision=8,suppress=False, threshold=1000, formatter=None)
#import scipy
##############################################################################
### 2. Define functions



##############################################################################
### 3. Import Images

os.chdir('C:\\Users\\grossar\\Box\\Incucyte\\Andrew')
os.chdir('C:\\Users\\grossar\\Box\\Incucyte\\Andrew\\2021y07m11d\\05h02m')
os.chdir('E:\\Andrew\\Incucyte')

imageFiles = os.listdir()
currentImageFile = imageFiles[4]
img=mpimg.imread(currentImageFile)
io.imshow(img)
"""
##############################################################################
### 4. Pre-processing
###### 4.1 - Parse image files   #############################################
########## 4.1.1 - Create metadata table
prefix     = []
wellNumber = []
posNumber  = []
dateValue  = []
timeValue  = []

for fileName in imageFiles:
    splitName = fileName.split('.')[0]
    splitName = splitName.split('_')
    prefix     += [splitName[0]]
    wellNumber += [splitName[1]]
    posNumber  += [splitName[2]]
    dateValue  += [splitName[3]]
    timeValue  += [splitName[4]]
    
imageMetadata = pd.DataFrame(list(zip(imageFiles, prefix, wellNumber, posNumber, dateValue, timeValue)), columns = ['imageFiles', 'prefix', 'wellNumber', 'posNumber', 'dateValue', 'timeValue'])

########## 4.1.2 - Subset metadata by project, date, and time
projectDict = {}                                      # Create a list of unique prefixes
wellDict = {}

for project in projectNames:                                                    # Loop through prefixes

    rowstoInclude = list(np.where(np.array(prefix) == project)[0])           # For each, identify their rows
    newDF = imageMetadata.iloc[rowstoInclude]
    dateNames = list(np.unique(newDF['dateValue']))
    dateDict = {}
    
    for dateValue in dateNames:
        rowstoInclude2 = list(np.where(np.array(newDF['dateValue']) == dateValue)[0])
        newDF2 = newDF.iloc[rowstoInclude2,:]
        timeNames = list(np.unique(newDF2['timeValue']))
        timeDict = {}
        
        for timeValue in timeNames:
            rowstoInclude2 = list(np.where(np.array(newDF2['timeValue']) == timeValue)[0])
            newDF3 = newDF2.iloc[rowstoInclude2,:]
            timeDict[timeValue] = newDF3
            wellNames = list(np.unique(newDF3['wellNumber']))

            for well in wellNames:
                rowstoInclude2 = list(np.where(np.array(newDF3['wellNumber']) == well)[0])
                wellList = list(newDF3.iloc[rowstoInclude2, 0])
                wellDict[project + '_' + dateValue + '_' + timeValue + '_' + well] = wellList
        
        dateDict[dateValue] = timeDict
        
    projectDict[project] = dateDict                                               # Add it to the DF list

### 

for well in list(wellDict):
    print('\n' + well + ':\n')
    for fileName in wellDict[well]:
        print(fileName)
    print('\n')

for currentImageFile in fileList:
    # Identify its well position
    
list(projectDict)     ## Returns a list of project names
list(projectDict.values())  ## Returns a list of dataframes
list(projectDict.values())[0].loc('dateValue')
                           
wellList = np.unique(wellNumber)

###### 4.2 - Stitch well images  #############################################
    indexes = np.where(np.array(ints) == item)[0]
    
for well in wellList:
    print(well)
    posToInclude = np.where(np.array(wellNumber) == well)[0].tolist()
    imagesToInclude = imageMetadata.iloc[posToInclude,:].iloc[:,0].tolist()
    for fileName in imagesToInclude:
        print(fileName)

img=mpimg.imread(currentImageFile)
"""
###### 4.2 - Rescale  #############################################


##############################################################################
### 4. Processing
###### 5.1 - Isolate mask  ###################################################

np.shape(img)
#img = rgb2hsv(img)[:, :, 1]   # Extract the saturation layer

img = closing(img > 0.5, square(9))

# remove artifacts connected to image border
s_cleared = clear_border(img)

# label image regions
label_img = label(s_cleared)
label_img = label(img)

# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`

s_img = img
img_label_overlay = label2rgb(label_img, image=s_img, bg_label=0)
io.imshow(img_label_overlay)

# Compile the properties of each region
img_data_l = regionprops(label_image = label_img, intensity_image = img)
img_data_df = regionprops_table(label_image = label_img, intensity_image = img, 
                             properties = ('label', 'centroid', 'convex_image', 'area', 'eccentricity', 'equivalent_diameter', 'euler_number'))

# List the number of regions
len(img_data_l)

img_data = pd.DataFrame(img_data_df)

img_data.head()
img_data.info

img_data.iloc[:,1]
img_data.iloc[1]
img_data.iloc[1,][3]

img_data_l[1].convex_image

'''
new = np.ones(np.shape(img)[0:2])
new[mask] = 0
io.imshow(new)
'''




###### 5.2 - Define colony list  #############################################


###### 5.3 - Convert colonies to polygons ####################################
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

coords_all_x = coords_all[:,0]
coords_all_y = coords_all[:,1]
coords_all_c = coords_all[:,2]

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

###### 5.5 - Create a matrix of points frome shared colonies  ################
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

print('For file ' + str(currentImageFile) + ' : ')
print('The median distance between colonies is ' + str(round(np.median(min_dist),1)) + ' um')
print('The minimum distance between colonies is ' + str(round(np.min(min_dist),1)) + ' um')
print('The maximum distance between colonies is ' + str(round(np.max(min_dist),1)) + ' um')
print('Two-thirds of colonies are between ' + str(round(np.median(min_dist)-np.std(min_dist))) + ' um and ' + 
     str(round(np.median(min_dist)+np.std(min_dist))) + ' um from neighboring colonies.')

