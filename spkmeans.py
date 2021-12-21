import numpy as np
import pandas as pd
from sys import argv
import re
import spkmeansmodule


def spectral_clustering(k: int, goal: str, data: np.ndarray):
    np.random.seed(0)
    num_points = data.shape[0]

    if k >= num_points:
        exit("Invalid Input!")

    # Calculating W
    weighted_adjacency_matrix = spkmeansmodule.calculate_W(data.tolist())
    if goal == 'wam':
        print_matrix(weighted_adjacency_matrix)
        return

    # Calculating D
    diagonal_degree_matrix = spkmeansmodule.calculate_D(
        weighted_adjacency_matrix)
    if goal == 'ddg':
        print_matrix(diagonal_degree_matrix)
        return

    # Calculating D^-0.5
    for i in range(num_points):
        diagonal_degree_matrix[i][i] **= -0.5

    # Calculating Lnorm
    lnorm = spkmeansmodule.calculate_Lnorm(
        weighted_adjacency_matrix, diagonal_degree_matrix)
    if goal == 'lnorm':
        print_matrix(lnorm)
        return

    # Finding eigenvalues and eigenvectors
    vals, vectors = perform_jacobi(np.array(lnorm))

    # Calculating the eigengap heuristic if needed
    if k == 0:
        delta = [np.abs(vals[i] - vals[i+1]) for i in range(num_points//2)]
        k = np.argmax(delta) + 1

    # Normalizing
    vectors = np.array(vectors[:, :k])
    for i, vector in enumerate(vectors):
        norm = np.linalg.norm(vector)
        if norm != 0:
            vectors[i] /= norm

    # Calling kmeans clustering algorithm
    centers, indices = kmeans(vectors, k)
    print(*indices, sep=',')
    print_matrix(centers)

# Performing kmeans++ for finding initial centers (As in exercise 2)
# and then calling the spkmeansmodule for performing the kmeans clustering algorithm
def kmeans(data: np.ndarray, k: int):
    np.random.seed(0)
    
    indices = [np.random.choice(data.shape[0] - 1)]
    centers = np.array([data[indices[0], :]])
    distances_from_centers = np.zeros((k, data.shape[0]))
    for i in range(k-1):
        for j, point in enumerate(data):
            distances_from_centers[i][j] = np.sum(
                np.square(point - centers[i]))

        min_weights = distances_from_centers[0:i+1].min(axis=0)
        prob_weights = min_weights / np.sum(min_weights)
        index = np.random.choice(data.shape[0], p=prob_weights)
        centers = np.vstack((centers, data[index]))
        indices.append(index)

    centers = spkmeansmodule.kmeans(data.tolist(), centers.tolist())
    return (centers, indices)

# Calling jacobi algorithm and sorting the eigenvalues and eigenvectors
def perform_jacobi(mat: np.ndarray):
    vals, vectors = spkmeansmodule.calculate_eigenvalues(mat.tolist())
    
    # Converting vectors to numpy array will help us treat it as a matrix
    vectors = np.array(vectors)
    
    # A dictionary that connects each eigenvalue to the fitting eigenvector
    val_to_vector = {vals[i]: vectors[:, i]
                     for i in range(mat.shape[0])}
    vals.sort()
    

    sorted_eigenvectors = np.array(val_to_vector[vals[0]])
    for val in vals[1:]:
        sorted_eigenvectors = np.column_stack(
            (sorted_eigenvectors, val_to_vector[val]))
    return vals, sorted_eigenvectors

# Printing a matrix in the requested format
def print_matrix(mat: np.ndarray):
    for i,row in enumerate(mat):
        print(*[f'{(val if np.abs(val) >= 1e-4 else 0):.4f}' for val in np.round(row, decimals=4)], sep=',',
                end='' if i == len(mat) - 1 else '\n')


if __name__ == '__main__':

    # If less than 3 arguments were given
    if len(argv) < 4:
        exit('Invalid Input!')

    # Checking if k is negative
    try:
        k = int(argv[1])
        if k < 0:
            exit('Invalid Input!')
    except ValueError:
        exit('Invalid Input!')

    # Checking if the second argument is a goal of the specified words
    goal = argv[2]
    if not re.fullmatch(string=goal, pattern='^(spk|wam|ddg|lnorm|jacobi)$'):
        exit('Invalid Input!')
    
    # Checking if the third argument is a legitimate path to a txt/csv file
    path = argv[3]
    if not re.fullmatch(string=path, pattern='^.+\.(txt|csv)$'):
        exit('Invalid Input!')

    # Reading the data and catching exceptions (In case the file doesn't exist or there is no data)
    try:
        data = pd.read_csv(path, header=None).to_numpy()
    except FileNotFoundError:
        exit("Invalid Input!")
    except pd.errors.EmptyDataError:
        exit("Invalid Input!")
    except pd.errors.ParserError:
        exit("An Error Has Occurred!")
    except Exception:
        exit("An Error Has Occurred!")

    # In case the goal is jacobi, the data from the file is a symmetrical matrix that we need
    # to run Jacobi's algorithm on. So we call the spkmeansmodule method and printing the values and vectors
    if goal == 'jacobi':
        vals, vectors = spkmeansmodule.calculate_eigenvalues(data.tolist())
        print(*[f'{i:.4f}' for i in vals], sep=',')
        print_matrix(np.array(vectors).T)

    else:
        spectral_clustering(k, goal, data)
