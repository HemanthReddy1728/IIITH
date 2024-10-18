import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2

# Load an image
image_path = './sample5.jpeg'
image = cv2.imread(image_path)

# Convert BGR image to RGB for Matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis("off")  # Turn off axes


def del_sq(a,b):
    return np.sum((a-b)**2)

def cal_energy(image):
    height, width= image.shape[:-1]
    energy = []
    for h in range(height):
        temp = []
        for w in range(width):
            next_h = (h+1)%height
            prev_h = (h-1)%height

            next_w = (w+1)%width
            prev_w = (w-1)%width

            del_sq_h = del_sq(image[next_h][w],image[prev_h][w])
            del_sq_w = del_sq(image[h][next_w],image[h][prev_w])

            temp.append(np.sqrt(del_sq_h + del_sq_w))
        energy.append(temp)
    
    return energy
            
energy = cal_energy(image)

def findVerticalSeam(energy):
    height = len(energy)
    width = len(energy[0])
    ver_ener = copy.deepcopy(energy)#energy.copy()
    for h in range(height-2,-1,-1):
        for w in range(width):
            if w!=0 and w!=width-1:
                ver_ener[h][w] += min(ver_ener[h+1][w-1],ver_ener[h+1][w],ver_ener[h+1][w+1])
            elif w==0:
                ver_ener[h][w] += min(ver_ener[h+1][w],ver_ener[h+1][w+1])
            else:
                ver_ener[h][w] += min(ver_ener[h+1][w-1],ver_ener[h+1][w])

    rem_coord = [ver_ener[0].index(min(ver_ener[0]))]
    
    for h in range(1,height):
        if rem_coord[-1] != 0 and rem_coord[-1] != width - 1:
            next_w = min(rem_coord[-1] - 1, rem_coord[-1], rem_coord[-1] + 1, key=lambda wt: ver_ener[h][wt])
            rem_coord.append(next_w)
        elif rem_coord[-1] == 0:
            next_w = min(rem_coord[-1], rem_coord[-1] + 1, key=lambda wt: ver_ener[h][wt])
            rem_coord.append(next_w)
        else:
            next_w = min(rem_coord[-1] - 1, rem_coord[-1], key=lambda wt: ver_ener[h][wt])
            rem_coord.append(next_w)
            
    return rem_coord

def findHorizontalSeam(energy):
    height = len(energy)
    width = len(energy[0])
    hor_ener = copy.deepcopy(energy)#energy.copy()
    for w in range(width-2,-1,-1):
        for h in range(height-1,-1,-1):
            if h!=0 and h!=height-1:
                hor_ener[h][w] += min(hor_ener[h-1][w+1],hor_ener[h][w+1],hor_ener[h+1][w+1])
            elif h==0:
                hor_ener[h][w] += min(hor_ener[h][w+1],hor_ener[h+1][w+1])
            else:
                hor_ener[h][w] += min(hor_ener[h-1][w+1],hor_ener[h][w+1])
        
    rem_coord = [min(range(height),key = lambda row_index: hor_ener[row_index][0])]

    for w in range(1,width):
        if rem_coord[-1]!=0 and rem_coord[-1]!= height-1:
            next_h = min(rem_coord[-1] - 1, rem_coord[-1], rem_coord[-1] + 1, key = lambda ht: hor_ener[ht][w])
            rem_coord.append(next_h)
        elif rem_coord[-1]==0:
            next_h = min(rem_coord[-1], rem_coord[-1] + 1, key = lambda ht: hor_ener[ht][w])
            rem_coord.append(next_h)
        else:
            next_h = min(rem_coord[-1] - 1, rem_coord[-1], key = lambda ht: hor_ener[ht][w])
            rem_coord.append(next_h)
    
    return rem_coord

def removeHorizontalSeam(image,rem_coord):
    height,width = image.shape[:-1]
    coord_del = [(i,elem) for i,elem in enumerate(rem_coordw)]
    mask = np.ones((height, width), dtype=bool)
    for h, w in coord_del:
        mask[h, w] = False
    new_image = image[mask]
    new_image = new_image.reshape(height,width-1,3)
    return new_image

def removeVerticalSeam(image,rem_coord):
    height,width = image.shape[:-1]
    coord_del = [(elem,i) for elem in enumerate(rem_coord)]
    mask = np.ones((height, width), dtype=bool)
    for h, w in coord_del:
        mask[h, w] = False
    new_image = image[mask]
    new_image = new_image.reshape(height,width-1,3)
    return new_image


#Trial run to remove 20 vertical seams
for i in range(200):
    energy = cal_energy(image)
    rem_coord = findVerticalSeam(energy)
    new_image = removeVerticalSeam(image,rem_coord)
    image = new_image
for i in range(200):
    energy = cal_energy(image)
    rem_coord = findHorizontalSeam(energy)
    new_image = removeHorizontalSeam(image,rem_coord)
    image = new_image


#Piece of code to display the new formed image
import matplotlib.pyplot as plt
# Convert BGR image to RGB for Matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis("off")  # Turn off axes
plt.show()
