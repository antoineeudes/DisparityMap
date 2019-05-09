import numpy as np
import scipy.ndimage
import imageio
import matplotlib.pyplot as plt


def compute_data_cost(I1, I2, num_disp_values, Tau):
    """data_cost: a 3D array of sixe height x width x num_disp_value;
    data_cost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    The cost is min(1/3 ||I1[y,x]-I2[y,x-l]||_1, Tau)."""
    h,w,_ = I1.shape
    dataCost=np.zeros((h,w,num_disp_values))
    for l in range(num_disp_values):
        dataCost[:,:,l] = np.minimum(np.linalg.norm((I1[:,:] - np.roll(I2, l, axis=1)), ord=1, axis=2)/3, Tau)
    return dataCost


def compute_energy(dataCost,disparity,Lambda):
    """dataCost: a 3D array of sixe height x width x num_disp_values;
    dataCost(y,x,l) is the cost of assigning the label l to pixel (y,x).
    disparity: array of size height x width containing disparity of each pixel.
    (an integer between 0 and num_disp_values-1)
    Lambda: a scalar value.
    Return total energy, a scalar value"""
    hh, ww = np.meshgrid(range(dataCost.shape[0]), range(dataCost.shape[1]), indexing='ij')
    dplp = dataCost[hh, ww, disparity]
    return np.sum(dplp) + Lambda * np.sum([disparity!=np.roll(disparity, 1, axis=0)] + [disparity!=np.roll(disparity, 1, axis=1)] + [disparity!=np.roll(disparity, -1, axis=0)] + [disparity!=np.roll(disparity, -1, axis=1)])

def update_msg(msgUPrev,msgDPrev,msgLPrev,msgRPrev,dataCost,Lambda):
    """Update message maps.
    dataCost: 3D array, depth=label number.
    msgUPrev,msgDPrev,msgLPrev,msgRPrev: 3D arrays (same dims) of old messages.
    Lambda: scalar value
    Return msgU,msgD,msgL,msgR: updated messages"""
    msgU=np.zeros(dataCost.shape)
    msgD=np.zeros(dataCost.shape)
    msgL=np.zeros(dataCost.shape)
    msgR=np.zeros(dataCost.shape)
    spqU = np.ones((dataCost.shape[0], dataCost.shape[1]))
    spqL = np.ones((dataCost.shape[0], dataCost.shape[1]))
    spqD = np.ones((dataCost.shape[0], dataCost.shape[1]))
    spqR = np.ones((dataCost.shape[0], dataCost.shape[1]))
    npqU = np.ones(dataCost.shape)
    npqL = np.ones(dataCost.shape)
    npqD = np.ones(dataCost.shape)
    npqR = np.ones(dataCost.shape)
    sum_msgU = np.roll(msgRPrev, -1, axis=0) + np.roll(msgLPrev, 1, axis=0) + np.roll(msgUPrev, 1, axis=1)
    sum_msgR = np.roll(msgRPrev, -1, axis=0) + np.roll(msgRPrev, 1, axis=0) + np.roll(msgUPrev, 1, axis=1)
    sum_msgL = np.roll(msgLPrev, 1, axis=0) + np.roll(msgDPrev, -1, axis=1) + np.roll(msgUPrev, 1, axis=1)
    sum_msgD = np.roll(msgRPrev, -1, axis=0) + np.roll(msgLPrev, 1, axis=0) + np.roll(msgDPrev, -1, axis=1)
    npqU = dataCost + sum_msgU
    npqL = dataCost + sum_msgL
    npqR = dataCost + sum_msgR
    npqD = dataCost + sum_msgD
    for l in range(dataCost.shape[2]):
        spqU[:,:] = np.minimum(spqU[:,:], npqU[:,:,l])
        spqL[:,:] = np.minimum(spqL[:,:], npqL[:,:,l])
        spqD[:,:] = np.minimum(spqD[:,:], npqD[:,:,l])
        spqR[:,:] = np.minimum(spqR[:,:], npqR[:,:,l])
    for l in range(dataCost.shape[2]):
        msgU[:,:,l] = np.minimum(dataCost[:,:,l] + sum_msgU[:,:,l], Lambda + spqU[:,:])
        msgL[:,:,l] = np.minimum(dataCost[:,:,l] + sum_msgL[:,:,l], Lambda + spqL[:,:])
        msgD[:,:,l] = np.minimum(dataCost[:,:,l] + sum_msgD[:,:,l], Lambda + spqD[:,:])
        msgR[:,:,l] = np.minimum(dataCost[:,:,l] + sum_msgR[:,:,l], Lambda + spqR[:,:])
    
    return msgU,msgD,msgL,msgR

def normalize_msg(msgU,msgD,msgL,msgR):
    """Subtract mean along depth dimension from each message"""
    avg=np.mean(msgU,axis=2)
    msgU -= avg[:,:,np.newaxis]
    avg=np.mean(msgD,axis=2)
    msgD -= avg[:,:,np.newaxis]
    avg=np.mean(msgL,axis=2)
    msgL -= avg[:,:,np.newaxis]
    avg=np.mean(msgR,axis=2)
    msgR -= avg[:,:,np.newaxis]
    return msgU,msgD,msgL,msgR

def compute_belief(dataCost,msgU,msgD,msgL,msgR):
    """Compute beliefs, sum of data cost and messages from all neighbors"""
    beliefs=dataCost.copy()
    beliefs += np.roll(msgR, -1, axis=0) + np.roll(msgL, 1, axis=0) + np.roll(msgU, 1, axis=1) + np.roll(msgD, -1, axis=1)
    return beliefs

def MAP_labeling(beliefs):
    """Return a 2D array assigning to each pixel its best label from beliefs
    computed so far"""
    return np.argmin(beliefs, axis=2)

def stereo_bp(I1,I2,num_disp_values,Lambda,Tau=15,num_iterations=60):
    """The main function"""
    dataCost = compute_data_cost(I1, I2, num_disp_values, Tau)
    energy = np.zeros((num_iterations)) # storing energy at each iteration
    # The messages sent to neighbors in each direction (up,down,left,right)
    h,w,_ = I1.shape
    msgU=np.zeros((h, w, num_disp_values))
    msgD=np.zeros((h, w, num_disp_values))
    msgL=np.zeros((h, w, num_disp_values))
    msgR=np.zeros((h, w, num_disp_values))

    for iter in range(num_iterations):
        msgU,msgD,msgL,msgR = update_msg(msgU,msgD,msgL,msgR,dataCost,Lambda)
        msgU,msgD,msgL,msgR = normalize_msg(msgU,msgD,msgL,msgR)
        # Next lines unused for next iteration, could be done only at the end
        beliefs = compute_belief(dataCost,msgU,msgD,msgL,msgR)
        disparity = MAP_labeling(beliefs)
        energy[iter] = compute_energy(dataCost,disparity,Lambda)
    return disparity,energy

# Input
img_left =imageio.imread('imL.png')
img_right=imageio.imread('imR.png')

print("Computing dataCost...")
dataCost = compute_data_cost(img_left, img_right, 5, 0.1)
plt.subplot(121)
plt.imshow(img_left)
plt.subplot(122)
plt.imshow(img_right)
plt.show()

# Convert as float gray images
img_left=img_left.astype(float)
img_right=img_right.astype(float)

# Parameters
num_disp_values=16 # these images have disparity between 0 and 15. 
Lambda=1.0

# Gaussian filtering
I1=scipy.ndimage.filters.gaussian_filter(img_left, 0.6)
I2=scipy.ndimage.filters.gaussian_filter(img_right,0.6)
print("Computing disparity and energy...")
disparity,energy = stereo_bp(I1,I2,num_disp_values,Lambda)
imageio.imwrite('disparity_{:g}.png'.format(Lambda),disparity)

# Plot results
plt.subplot(121)
plt.plot(energy)
plt.subplot(122)
plt.imshow(disparity,cmap='gray',vmin=0,vmax=num_disp_values-1)
plt.show()
