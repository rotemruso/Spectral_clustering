#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MAX_CHARS_IN_LINE 2000
#define MAX_POINTS 50
#define MAX_FEATURES 10
#define KMEANS_MAX_ITER 300
#define JACOBI_MAX_ITER 100
#define DELIMETER ","
#define EPSILON 1e-15
#define sign(x) (x < 0 ? -1 : 1)

#ifndef SPKMEANS_H
#define SPKMEANS_H
double **calculate_weighted_adjency_matrix(double **, int, int);
double **calculate_diagonal_degree_matrix(double **, int);
double **calculate_lnorm(double **, double **, int);
double** calculate_eigenvalues(double**,int);
double** kmeans_clustering(double** data,double** centers, int k, int num_points, int num_coordinates);
double **two_dimensional_alloc(int, int);
void free_matrix(double **);
#endif