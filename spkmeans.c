#include "spkmeans.h"

/* All the functions needed for the spectral clustering */
double **calculate_weighted_adjency_matrix(double **, int, int);
double **calculate_diagonal_degree_matrix(double **, int);
double **calculate_lnorm(double **, double **, int);
double **multiply_matrices(double **, double **, int);
void print_matrix(double **points, int rows, int columns);
double set_zero(double);
double norm_diff(double *, double *, int);
int spectral_clustering(int, char *, char *);
void power_minus_half(double **, int);
void minus_eye(double **, int);
double **read_points_from_file(FILE *, int, int);
int count_rows_in_file(FILE *);
int count_features_in_row(char *);
int eigengap_heuristic(double *, int);
void normalize(double **, int, int);

/* All the functions and variables needed for eigengap heuristic */
double eigenvalues[MAX_POINTS];
int compare_eigenvalues(const void *, const void *);
double **sort_eigenvalues_and_eigenvectors(double **, double **, int);
int *create_indices_array(int);

/* All the functions needed for Jacobi's algorithm */
int perform_jacobi(char *);
void transpose(double **, int);
double **eye(int);
double **create_spin_matrix(double, double, int, int, int);
double offset(double **, int);
void calculate_a_tag(double **, double **, double, double, int, int, int);

/* All the functions needed for kmeans clustering */
void add_points(double *, double *, int);
int find_closest_center(double *, double **, int, int);
void calc_avg(int, double *, int);
void set_to_zeros_clusters(int *, int);
void set_to_zeros_sums(double **, int k, int);
int check_if_converged(double **, double **, int, int);
void assign_mat_to_mat_coordinate_coordinate(double **, double **, int, int);

int main(int argc, char *argv[])
{
    int k;
    char *goal, *path;

    if (argc < 4)
    {
        printf("Invalid Input!\n");
        return EXIT_FAILURE;
    }
    k = atoi(argv[1]);
    goal = argv[2];

    /* Checking that goal is in [spk,wam,ddg,lnorm,jacobi] */
    if (strcmp(goal, "spk") && strcmp(goal, "wam") && strcmp(goal, "ddg") && strcmp(goal, "lnorm") &&
        strcmp(goal, "jacobi"))
    {
        printf("Invalid Input!\n");
        return EXIT_FAILURE;
    }
    path = argv[3];

    /* if goal == "jacobi" */
    if (!strcmp(goal, "jacobi"))
    {
        return perform_jacobi(path);
    }

    return spectral_clustering(k, goal, path);
}

int perform_jacobi(char *path)
{
    FILE *file;
    int size;
    int i;
    double **matrix, **eigenvectors;

    /* Reading the matrix */
    file = fopen(path, "r");
    if (!file)
    {
        printf("Invalid Input!\n");
        return EXIT_FAILURE;
    }
    size = count_rows_in_file(file);
    matrix = read_points_from_file(file, size, size);
    fclose(file);
    if (!matrix)
    {
        printf("An Error Has Occured!\n");
        free_matrix(matrix);
        return EXIT_FAILURE;
    }
    eigenvectors = calculate_eigenvalues(matrix, size);

    /* Printing the eigenvalues */
    for (i = 0; i < size - 1; i++)
    {
        printf("%.4f,", set_zero(matrix[i][i]));
    }
    printf("%.4f\n", set_zero(matrix[i][i]));

    /* Transposing the sorted eigenvectors matrix so that the rows will be the eigenvectors */
    transpose(eigenvectors, size);
    print_matrix(eigenvectors, size, size);

    free_matrix(eigenvectors);
    free_matrix(matrix);
    return EXIT_SUCCESS;
}

void transpose(double **matrix, int size)
{
    int i, j;
    double temp;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < i; j++)
        {
            temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

/* Performing Spectral Clustering and returning 0 if succeeded and 1 if something failed */
int spectral_clustering(int k, char *goal, char *path)
{
    double **data;
    double **W, **D, **lnorm;
    double **eigenvectors;
    double **sorted_eigenvectors;
    double **centers;
    double **new_centers;
    int num_points, num_coordinates;
    FILE *file;
    char line[MAX_CHARS_IN_LINE];

    file = fopen(path, "r");
    if (!file)
    {
        printf("Invalid Input!\n");
        return EXIT_FAILURE;
    }
    num_points = count_rows_in_file(file);
    if (k >= num_points)
    {
        printf("Invalid Input!");
        return EXIT_FAILURE;
    }
    fscanf(file, "%s", line);
    rewind(file);
    num_coordinates = count_features_in_row(line);
    data = read_points_from_file(file, num_points, num_coordinates);
    if (data == NULL)
    {
        printf("An Error Has Occured!\n");
        return EXIT_FAILURE;
    }
    fclose(file);

    W = calculate_weighted_adjency_matrix(data, num_points, num_coordinates);
    free_matrix(data);
    if (!strcmp(goal, "wam"))
    {
        print_matrix(W, num_points, num_points);
        free_matrix(W);
        return EXIT_SUCCESS;
    }

    D = calculate_diagonal_degree_matrix(W, num_points);

    if (!strcmp(goal, "ddg"))
    {
        print_matrix(D, num_points, num_points);
        free_matrix(W);
        free_matrix(D);
        return EXIT_SUCCESS;
    }

    power_minus_half(D, num_points);
    lnorm = calculate_lnorm(W, D, num_points);
    free_matrix(W);
    free_matrix(D);
    if (!strcmp(goal, "lnorm"))
    {
        print_matrix(lnorm, num_points, num_points);
        free_matrix(lnorm);
        return EXIT_SUCCESS;
    }

    eigenvectors = calculate_eigenvalues(lnorm, num_points);
    sorted_eigenvectors = sort_eigenvalues_and_eigenvectors(lnorm, eigenvectors, num_points);
    free_matrix(eigenvectors);
    free_matrix(lnorm);
    if (k == 0)
    {
        k = eigengap_heuristic(eigenvalues, num_points);
    }
    normalize(sorted_eigenvectors, num_points, k);
    
    /* Selecting the first k points to be the initial centers */
    centers = two_dimensional_alloc(k, k);
    assign_mat_to_mat_coordinate_coordinate(centers, sorted_eigenvectors, k, k);
    new_centers = kmeans_clustering(sorted_eigenvectors, centers, k, num_points, k);

    print_matrix(new_centers, k, k);
    free_matrix(new_centers);
    free_matrix(centers);
    free_matrix(sorted_eigenvectors);
    return EXIT_SUCCESS;
}

/* Performing k-means clustering, same as ex1 and ex2 */
double **kmeans_clustering(double **data, double **centers, int k, int num_points, int num_coordinates)
{
    int i;
    double **prev_centers = two_dimensional_alloc(k, num_coordinates);
    double **sum = two_dimensional_alloc(k, num_coordinates);
    int iter;
    int *clusters = calloc(k, sizeof(int));
    assert(clusters && "An Error Has Occurred!");

    for (iter = 0; iter < KMEANS_MAX_ITER; iter++)
    {

        for (i = 0; i < num_points; i++)
        {
            int closest_center = find_closest_center(data[i], centers, num_coordinates, k);

            clusters[closest_center]++;
            add_points(sum[closest_center], data[i], num_coordinates);
        }

        for (i = 0; i < k; i++)
        {
            calc_avg(clusters[i], sum[i], num_coordinates);
        }
        assign_mat_to_mat_coordinate_coordinate(centers, sum, k, num_coordinates);

        if (check_if_converged(centers, prev_centers, k, num_coordinates))
        {
            break;
        }
        set_to_zeros_clusters(clusters, k);
        set_to_zeros_sums(sum, k, num_coordinates);
        assign_mat_to_mat_coordinate_coordinate(prev_centers, centers, k, num_coordinates);
    }

    free(clusters);
    free_matrix(sum);
    return prev_centers;
}

/* normalizing each row (dividing by its L2 norm) */
void normalize(double **matrix, int rows, int columns)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        double norm = 0;
        for (j = 0; j < columns; j++)
        {
            norm += matrix[i][j] * matrix[i][j];
        }
        norm = sqrt(norm);
        if (norm != 0)
        {
            for (j = 0; j < columns; j++)
            {
                matrix[i][j] /= norm;
            }
        }
    }
}

/* Adding point2 to point1 */
void add_points(double *point1, double *point2, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        point1[i] += point2[i];
    }
}

/* Passing mat2's values to mat1 */
void assign_mat_to_mat_coordinate_coordinate(double **mat1, double **mat2, int rows, int columns)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            mat1[i][j] = mat2[i][j];
        }
    }
}

/* Check if the difference between the new centers and old centers is 0 */
int check_if_converged(double **centers, double **prev_centers, int k, int size)
{
    int i, j;

    for (i = 0; i < k; i++)
    {
        for (j = 0; j < size; j++)
        {
            if (centers[i][j] - prev_centers[i][j])
            {
                return 0;
            }
        }
    }

    return 1;
}

/* Resetting the clusters array's values to 0 */
void set_to_zeros_clusters(int clusters[], int k)
{
    int i;

    for (i = 0; i < k; i++)
    {
        clusters[i] = 0;
    }
}

/* Resetting the sums matrix to 0 */
void set_to_zeros_sums(double **sums, int k, int size)
{
    int i, j;

    for (i = 0; i < k; i++)
    {
        for (j = 0; j < size; j++)
        {
            sums[i][j] = 0;
        }
    }
}

/* Finding the closest center to a given point */
int find_closest_center(double *point, double **centers, int size, int k)
{
    int min_index = 0;
    double min_norm;
    double curr_norm;
    int i = 1;

    min_norm = norm_diff(centers[0], point, size);
    for (; i < k; i++)
    {
        curr_norm = norm_diff(centers[i], point, size);
        if (curr_norm < min_norm)
        {
            min_norm = curr_norm;
            min_index = i;
        }
    }

    return min_index;
}

/* Calculating the average of the points in a cluster to determine the new center */
void calc_avg(int cluster, double *sum, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        sum[i] /= (cluster != 0 ? cluster : 1);
    }
}

/* Read the data from a given file. We assume that the data in each row is real numbers with 4
   digits after the decimal point, separated by commas */
double **read_points_from_file(FILE *file, int num_points, int num_features)
{
    double val;
    double **points = two_dimensional_alloc(num_points, num_features);
    char *strval, line[MAX_CHARS_IN_LINE];
    int i = 0, j = 0;
    while (fscanf(file, "%s", line) != EOF)
    {
        strval = strtok(line, DELIMETER);
        while (strval != NULL)
        {
            val = atof(strval);
            points[i][j] = val;
            j++;
            strval = strtok(NULL, DELIMETER);
        }
        j = 0;
        i++;
    }
    return points;
}

/* Counting how many features/coordinates are in a row for determining how many columns to allocate */
int count_features_in_row(char *line)
{
    int count = 1;
    size_t len = strlen(line);
    size_t i;
    for (i = 0; i < len; i++)
    {
        if (line[i] == DELIMETER[0])
        {
            count++;
        }
    }
    return count;
}

/* Counting how many rows are in a file for allocating the exact number of rows */
int count_rows_in_file(FILE *file)
{
    int count = 0;
    char line[MAX_CHARS_IN_LINE];

    while (fscanf(file, "%s", line) != EOF)
    {
        count++;
    }
    rewind(file);
    return count;
}

/* Calculating the Lnorm matrix */
double **calculate_lnorm(double **weighted_adjacency_matrix, double **diagonal_degree_matrix_normalized, int size)
{
    double **mul1 = multiply_matrices(diagonal_degree_matrix_normalized, weighted_adjacency_matrix, size);
    double **mul = multiply_matrices(mul1, diagonal_degree_matrix_normalized, size);
    free_matrix(mul1);
    minus_eye(mul, size);
    return mul;
}

/* Calculating D^-0.5 */
void power_minus_half(double **matrix, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        matrix[i][i] = pow(matrix[i][i], -0.5);
    }
}

/* Returning the result of I - matrix when I is the unit matrix */
void minus_eye(double **matrix, int size)
{
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            matrix[i][j] = (i == j ? 1 : 0) - matrix[i][j];
        }
    }
}

/* Performing matrix multiplication between two square matrices with the same dimensions */
double **multiply_matrices(double **mat1, double **mat2, int size)
{
    double **mul = two_dimensional_alloc(size, size);
    int i, j, k;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            for (k = 0; k < size; k++)
            {
                mul[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mul;
}

/* Calculating the matrix D */
double **calculate_diagonal_degree_matrix(double **weighted_adjency_matrix, int size)
{
    double **diagonal_degree_matrix = two_dimensional_alloc(size, size);
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            diagonal_degree_matrix[i][i] += weighted_adjency_matrix[i][j];
        }
    }

    return diagonal_degree_matrix;
}

/* From Lecture 4, slide 15 - Allocating two dimensional array in one contiguous block */
double **two_dimensional_alloc(int rows, int columns)
{
    double **arr_to_return;
    double *p;
    int i;

    p = calloc(rows * columns, sizeof(double));
    assert(p && "An Error Has Occurred!");
    arr_to_return = calloc(rows, sizeof(double *));
    assert(arr_to_return && "An Error Has Occurred!");
    for (i = 0; i < rows; i++)
    {
        arr_to_return[i] = p + i * columns;
    }

    return arr_to_return;
}

/* Calculating a norm of the difference of two vectors */
double norm_diff(double *point1, double *point2, int size)
{
    double res = 0;
    int i;

    for (i = 0; i < size; i++)
    {
        res += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    return res;
}

/* Calculating W */
double **calculate_weighted_adjency_matrix(double **points, int num_points, int num_coordinates)
{
    double **weighted_adjency_matrix = two_dimensional_alloc(num_points, num_points);
    int i, j;

    for (i = 0; i < num_points; i++)
    {
        for (j = 0; j < num_points; j++)
        {
            weighted_adjency_matrix[i][j] = i == j ? 0.0 : exp(-sqrt(norm_diff(points[i], points[j], num_coordinates)) / 2.0);
        }
    }

    return weighted_adjency_matrix;
}

/* A method made for printing matrices in the required format */
void print_matrix(double **points, int rows, int columns)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns - 1; j++)
        {
            printf("%.4f,", set_zero(points[i][j]));
        }
        printf("%.4f%s", set_zero(points[i][columns - 1]), i == rows - 1 ? "\0" : "\n");
    }
}

/* returns 0 if the number is -0.000 in the printing format .4f */
double set_zero(double number)
{
    char data[10];

    sprintf(data, "%.4f", fabs(number));
    return strcmp(data, "0.0000") ? number : 0;
}

/* Jacobi's algorithm. It "spins" the matrix A such that A will eventually be diagonal with the
   eigenvalues and it returns the eigenvectors themselves (multiplication of the spinning matrices)
 */
double **calculate_eigenvalues(double **A, int size)
{
    double convergeance = 0;
    int num_iter = 0;
    int i = 0, i_max = 0;
    int j = 0, j_max = 0;
    double **A_tag;
    double theta, t, c, s;
    double a_offset, a_tag_offset;
    double **P;
    double **multiplication;
    double **V = eye(size);

    A_tag = two_dimensional_alloc(size, size);

    for (; num_iter < JACOBI_MAX_ITER; num_iter++)
    {
        /* Calculate max off diagonal */
        double max = -1;

        for (i = 0; i < size; i++)
        {
            for (j = 0; j < i; j++)
            {
                double ab_val = fabs(A[i][j]);
                if (ab_val > max)
                {
                    max = ab_val;
                    i_max = i;
                    j_max = j;
                }
            }
        }
        theta = (A[j_max][j_max] - A[i_max][i_max]) / (2 * A[i_max][j_max]);
        t = sign(theta) / (fabs(theta) + sqrt(theta * theta + 1));
        c = pow(t * t + 1, -0.5);
        s = t * c;
        P = create_spin_matrix(c, s, i_max, j_max, size);
        multiplication = multiply_matrices(V, P, size);
        free_matrix(V);
        free_matrix(P);
        V = multiplication;

        a_offset = offset(A, size);
        assign_mat_to_mat_coordinate_coordinate(A_tag, A, size, size);
        calculate_a_tag(A, A_tag, c, s, i_max, j_max, size);
        a_tag_offset = offset(A_tag, size);
        convergeance = fabs(a_offset - a_tag_offset);
        if (convergeance <= EPSILON)
        {
            break;
        }
        assign_mat_to_mat_coordinate_coordinate(A, A_tag, size, size);
    }
    free_matrix(A_tag);
    return V;
}

/* 
   We sort the indices of the eigenvalues in order to place the eigenvectors in the right order.
   Since we use the method qsort we need a comparing function, and this comparing function
   compares the indices by comparing the eigenvalues in these indices.
*/
int compare_eigenvalues(const void *a, const void *b)
{
    double res = eigenvalues[*(int *)a] - eigenvalues[*(int *)b];
    if (res == 0)
    {
        return 0;
    }
    return sign(res);
}

/* 
   Since after jacobi's algorithm we have a diagonal matrix and a matrix of eigenvectors,
   we want to sort the vectors according to the values. Therefore, we sort the list of indices 
   such that the indices will show the permutation of the sorted list. Since each eigenvalue has
   a corresponding eigenvector, the indices permutation will fit them too
*/
double **sort_eigenvalues_and_eigenvectors(double **matrix, double **eigenvectors, int size)
{
    int i, j;
    int *indices;
    double **sorted_eigenvectors = two_dimensional_alloc(size, size);

    for (i = 0; i < size; i++)
    {
        eigenvalues[i] = matrix[i][i];
    }
    indices = create_indices_array(size);
    qsort(indices, size, sizeof(int), compare_eigenvalues);
    for (i = 0; i < size; i++)
    {
        eigenvalues[i] = matrix[indices[i]][indices[i]];
    }

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            sorted_eigenvectors[i][j] = eigenvectors[i][indices[j]];
        }
    }
    free(indices);
    return sorted_eigenvectors;
}

/* Creating an array of indices */
int *create_indices_array(int size)
{
    int i;
    int *indices = malloc(size * sizeof(int));
    assert(indices && "An Error Has Occurred!");

    for (i = 0; i < size; i++)
    {
        indices[i] = i;
    }
    return indices;
}

/* Creating the spin matrix P for Jacobi's algorithm */
double **create_spin_matrix(double c, double s, int i, int j, int size)
{
    double **P = eye(size);
    P[i][i] = c;
    P[j][j] = c;
    P[i][j] = s;
    P[j][i] = -s;
    return P;
}

/* Performing the multiplication P.transpose() @ A @ P */
void calculate_a_tag(double **A, double **A_tag, double c, double s, int i, int j, int size)
{
    int r;
    for (r = 0; r < size; r++)
    {
        if (r != i && r != j)
        {
            A_tag[r][i] = c * A[r][i] - s * A[r][j];
            A_tag[i][r] = A_tag[r][i];
            A_tag[r][j] = c * A[r][j] + s * A[r][i];
            A_tag[j][r] = A_tag[r][j];
        }
    }
    A_tag[i][i] = c * c * A[i][i] + s * s * A[j][j] - 2 * s * c * A[i][j];
    A_tag[j][j] = s * s * A[i][i] + c * c * A[j][j] + 2 * s * c * A[i][j];
    A_tag[i][j] = 0;
    A_tag[j][i] = 0;
}

/* Calculating the sum of squares of the off diagonal values */
double offset(double **matrix, int size)
{
    double sum = 0;
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            sum += i != j ? matrix[i][j] * matrix[i][j] : 0;
        }
    }
    return sum;
}

/* Creating a unit matrix */
double **eye(int size)
{
    double **mat = two_dimensional_alloc(size, size);
    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            mat[i][j] = i == j ? 1 : 0;
        }
    }
    return mat;
}

/* Calculating k from the sorted eigenvalues list */
int eigengap_heuristic(double *eigenvalues, int size)
{
    int k = 0;
    double max_diff = -1;
    int i;
    for (i = 0; i < size / 2; i++)
    {
        double curr_diff = fabs(eigenvalues[i] - eigenvalues[i + 1]);
        if (curr_diff > max_diff)
        {
            max_diff = curr_diff;
            k = i;
        }
    }
    return k + 1;
}

/* Freeing a matrix. First we free the block itself and then the given pointer */
void free_matrix(double **array)
{
    free(*array);
    free(array);
}