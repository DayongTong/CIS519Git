import numpy as np


# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    """
    Given a list of names of length N, return a numpy matrix of shape (N, 260)
    with the features described in problem 2b of the homework assignment.
    
    Parameters
    ----------
    names: A list of strings
        The names to featurize, e.g. ["albert einstein", "marie curie"]
    
    Returns
    -------
    numpy.array:
        A numpy array of shape (N, 260)
    """

    rows, cols = (len(names), 2) 
    txtarr = [[0]*cols]*rows 
    for i,row in enumerate(names):
        txtarr[i] = row.split()

    X = np.zeros([len(names),260])
    for i,row in enumerate(txtarr):
        string1 = txtarr[i][0]
        string2 = txtarr[i][1]
        feature_row = np.zeros(260,)
        for j in range(5):
            if j < len(string1):
                feature_row[j*26:(j+1)*26] = alphabet_loc(string1[j])
        for j in range(5):
            if j < len(string2):
                feature_row[(j+5)*26:(j+6)*26] = alphabet_loc(string2[j])
        X[i,:] = feature_row
    return X

    # raise NotImplementedError

def alphabet_loc(char):
#     returns the position of the character in the alphabet in a 1X26 of 0,1 array, eg 1 at array[3] means char is a C
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    index = alphabet.index(char)
    result = np.zeros([26,])
    result[index] = 1
    return result

def convert_X_txt(lines,txtarr):
    X = np.zeros([len(lines),260])
    for i,row in enumerate(txtarr):
        string1 = txtarr[i][0]
        string2 = txtarr[i][1]
        feature_row = np.zeros(260,)
        for j in range(5):
            if j < len(string1):
                feature_row[j*26:(j+1)*26] = alphabet_loc(string1[j])
        for j in range(5):
            if j < len(string2):
                feature_row[(j+5)*26:(j+6)*26] = alphabet_loc(string2[j])
        X[i,:] = feature_row
    return X