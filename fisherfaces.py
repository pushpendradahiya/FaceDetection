import numpy as np
from eigenfaces import get_images
from eigenfaces import Eigenface_model

def Fishers_model( images, labels):
    # Get the shape of image variable and the no of classes
    d = images.shape [1]
    classes = np.unique(labels)
    # Initailise the two covariance matrix, Sw- within cluster, Sw- Between clusters
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    # Get the mean of the total set
    totmean=images.mean(axis=0, keepdims=True)
    # calculate and update Sw and Sb
    for i in range(0, len(classes)):
        imagesi = images[np.where(labels == i+1)[0], :]
        ni = imagesi.shape[0]
        MEANi = imagesi.mean(axis=0, keepdims=True)
        Sw = Sw + np.matmul((imagesi - MEANi).T, (imagesi - MEANi))
        Sb = Sb + ni * np.matmul((MEANi - totmean).T, (MEANi - totmean))

    # Get the eigen values and eigen vectors of the matrix Inv(Sw)*Sb
    eigval_fld, eigvec_fld = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    # Sort eigen vectors and eigen values in decreasing order of the eigenvalues
    idx = np.argsort(-eigval_fld.real)
    eigval_fld = eigval_fld[idx].real
    eigvec_fld = eigvec_fld[:, idx].real
    eigval_fld = eigval_fld[0:(len(classes)-1)]
    eigvec_fld = eigvec_fld[:,0:(len(classes)-1)]
    return eigval_fld, eigvec_fld

def Fishers_test(images_train, images_test, labels_test, labels_train, eigvec, mean):
    # Get the weights of the trained images
    weights = np.matmul(eigvec.T, images_train)
    # Normalize the test images
    images_test = images_test.T - mean
    labels_test = labels_test.T

    # calculate test image weights
    testweights = np.matmul(eigvec.T, images_test)

    correct = 0
    # Calculate the error for each test image and find the closest training image and display the result
    for i in range(0, len(labels_test)):
        # calculate error for each test image
        testweight = np.resize(testweights[:, i], (testweights.shape[0], 1))
        err = (weights - testweight) ** 2
        #stddiv = np.std(weights, axis=0, keepdims=True)
        #err = err / stddiv

        # calculate the sum of square of error
        ssq1 = np.sum(err **(1/2), axis=0)

        # Find the closest face to the test image
        dist = ssq1.min(axis=0, keepdims=True)
        match = labels_train[ssq1.argmin(axis=0)]

        # print the subject number
        if dist < 50000:
            if labels_test[i] == match:
                correct+=1
                print("subject %d identified correctly as %d with distance %f" % (labels_test[i], match, dist.real))
            else:
                print("subject %d identified incorrectly as %d with distance %f" % (labels_test[i], match, dist.real))
        else:
            print("subject face not match in database")
    print("The accuracy of Fisherfaces is %f percent" %(correct*100/len(labels_test)))

if __name__ == "__main__":
    # Get the images and labels from path
    path = "E:\yalefaces1\yalefaces"
    images_train, labels_train = get_images(path)

    # Split the data into training and test
    path = "E:\yalefaces1\yalefacestest"
    images_test, labels_test = get_images(path)

    # Perform Eigenface analysis and get Eigenface vectors
    eigval_pca, eigvec_pca, mean= Eigenface_model(images_train, labels_train)

    [n,d]= images_train.shape
    c = len(np.unique(labels_train))
    # Reduce it to n-c dimension
    eigvec_pca1 = eigvec_pca[:,0: n-c]

    # Project the images onto n-c dimension using PCA
    images_train = images_train.T
    images_train = images_train - mean
    images_train_project = np.matmul(images_train.T, eigvec_pca1)

    # Perform Fishers linear discriminant on the projected images
    eigval_fld, eigvec_fld = Fishers_model( images_train_project, labels_train)

    # Get the total final eigen vectors by multiplying the eigen vectors from PCA and FLD
    eigvec= np.matmul(eigvec_pca1,eigvec_fld)

    # Test the Fisherface model on the test images and print the result
    Fishers_test(images_train, images_test, labels_test, labels_train, eigvec, mean)