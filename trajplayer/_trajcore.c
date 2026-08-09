#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    int64_t x;
    int64_t y;
    int64_t z;
    int32_t atom;
} CellEntry;


typedef struct {
    int64_t x;
    int64_t y;
    int64_t z;
    npy_intp start;
    npy_intp end;
} CellGroup;


typedef struct {
    float *distance2;
    int32_t *left;
    int32_t *right;
    size_t count;
    size_t capacity;
    int failed;
} CandidateBuffer;


static int compare_entries(const void *lhs, const void *rhs) {
    const CellEntry *a = (const CellEntry *)lhs;
    const CellEntry *b = (const CellEntry *)rhs;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    if (a->z != b->z) return a->z < b->z ? -1 : 1;
    if (a->atom != b->atom) return a->atom < b->atom ? -1 : 1;
    return 0;
}


static int compare_cell(
    int64_t x,
    int64_t y,
    int64_t z,
    const CellGroup *group
) {
    if (x != group->x) return x < group->x ? -1 : 1;
    if (y != group->y) return y < group->y ? -1 : 1;
    if (z != group->z) return z < group->z ? -1 : 1;
    return 0;
}


static npy_intp find_group(
    const CellGroup *groups,
    npy_intp group_count,
    int64_t x,
    int64_t y,
    int64_t z
) {
    npy_intp low = 0;
    npy_intp high = group_count;
    while (low < high) {
        npy_intp middle = low + (high - low) / 2;
        int comparison = compare_cell(x, y, z, &groups[middle]);
        if (comparison == 0) return middle;
        if (comparison < 0) high = middle;
        else low = middle + 1;
    }
    return -1;
}


static int inverse3x3(const double matrix[3][3], double inverse[3][3]) {
    double determinant =
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if (!isfinite(determinant) || fabs(determinant) <= 1.0e-12) return 0;
    double scale = 1.0 / determinant;
    inverse[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * scale;
    inverse[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * scale;
    inverse[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * scale;
    inverse[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * scale;
    inverse[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * scale;
    inverse[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * scale;
    inverse[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * scale;
    inverse[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * scale;
    inverse[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * scale;
    return 1;
}


static int32_t find_root(int32_t *parent, int32_t node) {
    int32_t root = node;
    while (parent[root] >= 0) root = parent[root];
    while (node != root) {
        int32_t next = parent[node];
        parent[node] = root;
        node = next;
    }
    return root;
}


static PyObject *trajcore_connected_components(PyObject *self, PyObject *args) {
    Py_ssize_t atom_count_arg;
    PyObject *bonds_object;
    (void)self;
    if (!PyArg_ParseTuple(args, "nO", &atom_count_arg, &bonds_object)) return NULL;
    if (atom_count_arg < 0 || atom_count_arg > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "atom_count must fit in int32");
        return NULL;
    }

    PyArrayObject *bonds = (PyArrayObject *)PyArray_FROM_OTF(
        bonds_object,
        NPY_INT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (bonds == NULL) return NULL;
    if (PyArray_NDIM(bonds) != 2 || PyArray_DIM(bonds, 1) != 2) {
        Py_DECREF(bonds);
        PyErr_SetString(PyExc_ValueError, "bonds must have shape (M, 2)");
        return NULL;
    }

    int32_t atom_count = (int32_t)atom_count_arg;
    npy_intp bond_count = PyArray_DIM(bonds, 0);
    int32_t *pairs = (int32_t *)PyArray_DATA(bonds);
    int32_t *parent = (int32_t *)malloc((size_t)atom_count * sizeof(int32_t));
    int32_t *root_labels = (int32_t *)malloc((size_t)atom_count * sizeof(int32_t));
    int32_t *sizes_temp = (int32_t *)calloc((size_t)atom_count, sizeof(int32_t));
    if ((atom_count > 0) && (parent == NULL || root_labels == NULL || sizes_temp == NULL)) {
        free(parent);
        free(root_labels);
        free(sizes_temp);
        Py_DECREF(bonds);
        return PyErr_NoMemory();
    }

    npy_intp label_dims[1] = {atom_count};
    PyArrayObject *labels = (PyArrayObject *)PyArray_SimpleNew(1, label_dims, NPY_INT32);
    if (labels == NULL) {
        free(parent);
        free(root_labels);
        free(sizes_temp);
        Py_DECREF(bonds);
        return NULL;
    }
    int32_t *label_data = (int32_t *)PyArray_DATA(labels);
    int invalid_pair = 0;
    int32_t component_count = 0;

    Py_BEGIN_ALLOW_THREADS
    for (int32_t i = 0; i < atom_count; ++i) {
        parent[i] = -1;
        root_labels[i] = -1;
    }
    for (npy_intp edge = 0; edge < bond_count; ++edge) {
        int32_t left = pairs[edge * 2];
        int32_t right = pairs[edge * 2 + 1];
        if (left < 0 || left >= atom_count || right < 0 || right >= atom_count) {
            invalid_pair = 1;
            break;
        }
        int32_t left_root = find_root(parent, left);
        int32_t right_root = find_root(parent, right);
        if (left_root == right_root) continue;
        if (parent[left_root] > parent[right_root]) {
            int32_t temporary = left_root;
            left_root = right_root;
            right_root = temporary;
        }
        parent[left_root] += parent[right_root];
        parent[right_root] = left_root;
    }
    if (!invalid_pair) {
        for (int32_t atom = 0; atom < atom_count; ++atom) {
            int32_t root = find_root(parent, atom);
            int32_t label = root_labels[root];
            if (label < 0) {
                label = component_count++;
                root_labels[root] = label;
            }
            label_data[atom] = label;
            sizes_temp[label] += 1;
        }
    }
    Py_END_ALLOW_THREADS

    free(parent);
    free(root_labels);
    Py_DECREF(bonds);
    if (invalid_pair) {
        free(sizes_temp);
        Py_DECREF(labels);
        PyErr_SetString(PyExc_ValueError, "bonds contains atom indices outside atom_count");
        return NULL;
    }

    npy_intp size_dims[1] = {component_count};
    PyArrayObject *sizes = (PyArrayObject *)PyArray_SimpleNew(1, size_dims, NPY_INT32);
    if (sizes == NULL) {
        free(sizes_temp);
        Py_DECREF(labels);
        return NULL;
    }
    if (component_count > 0) {
        memcpy(PyArray_DATA(sizes), sizes_temp, (size_t)component_count * sizeof(int32_t));
    }
    free(sizes_temp);
    return Py_BuildValue("NN", labels, sizes);
}


static int grow_candidates(CandidateBuffer *buffer) {
    size_t next_capacity = buffer->capacity == 0 ? 1024 : buffer->capacity * 2;
    if (next_capacity <= buffer->capacity || next_capacity > SIZE_MAX / sizeof(float)) {
        buffer->failed = 1;
        return 0;
    }
    float *next_distance2 = (float *)realloc(
        buffer->distance2,
        next_capacity * sizeof(float)
    );
    if (next_distance2 == NULL) {
        buffer->failed = 1;
        return 0;
    }
    buffer->distance2 = next_distance2;
    int32_t *next_left = (int32_t *)realloc(
        buffer->left,
        next_capacity * sizeof(int32_t)
    );
    if (next_left == NULL) {
        buffer->failed = 1;
        return 0;
    }
    buffer->left = next_left;
    int32_t *next_right = (int32_t *)realloc(
        buffer->right,
        next_capacity * sizeof(int32_t)
    );
    if (next_right == NULL) {
        buffer->failed = 1;
        return 0;
    }
    buffer->right = next_right;
    buffer->capacity = next_capacity;
    return 1;
}


static void append_candidate(
    CandidateBuffer *buffer,
    float distance2,
    int32_t left,
    int32_t right
) {
    if (buffer->failed) return;
    if (buffer->count == buffer->capacity && !grow_candidates(buffer)) return;
    size_t index = buffer->count++;
    buffer->distance2[index] = distance2;
    buffer->left[index] = left < right ? left : right;
    buffer->right[index] = left < right ? right : left;
}


static float pair_distance2(
    const float *positions,
    int32_t left,
    int32_t right,
    int periodic,
    const double cell[3][3],
    const double inverse[3][3]
) {
    double delta[3] = {
        (double)positions[(size_t)right * 3] - positions[(size_t)left * 3],
        (double)positions[(size_t)right * 3 + 1] - positions[(size_t)left * 3 + 1],
        (double)positions[(size_t)right * 3 + 2] - positions[(size_t)left * 3 + 2],
    };
    if (periodic) {
        double fractional[3];
        for (int axis = 0; axis < 3; ++axis) {
            fractional[axis] =
                delta[0] * inverse[0][axis]
                + delta[1] * inverse[1][axis]
                + delta[2] * inverse[2][axis];
            fractional[axis] -= nearbyint(fractional[axis]);
        }
        for (int axis = 0; axis < 3; ++axis) {
            delta[axis] =
                fractional[0] * cell[0][axis]
                + fractional[1] * cell[1][axis]
                + fractional[2] * cell[2][axis];
        }
    }
    return (float)(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
}


static int64_t wrap_cell(int64_t value, int64_t count) {
    int64_t wrapped = value % count;
    return wrapped < 0 ? wrapped + count : wrapped;
}


static PyObject *empty_candidates(void) {
    npy_intp dims[1] = {0};
    PyObject *distance2 = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *left = PyArray_SimpleNew(1, dims, NPY_INT32);
    PyObject *right = PyArray_SimpleNew(1, dims, NPY_INT32);
    if (distance2 == NULL || left == NULL || right == NULL) {
        Py_XDECREF(distance2);
        Py_XDECREF(left);
        Py_XDECREF(right);
        return NULL;
    }
    return Py_BuildValue("NNN", distance2, left, right);
}


static PyObject *trajcore_candidate_pairs(PyObject *self, PyObject *args) {
    PyObject *positions_object;
    PyObject *active_object;
    PyObject *cell_object = Py_None;
    double cutoff;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOd|O", &positions_object, &active_object, &cutoff, &cell_object)) {
        return NULL;
    }
    if (!isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "cutoff must be positive and finite");
        return NULL;
    }

    PyArrayObject *positions = (PyArrayObject *)PyArray_FROM_OTF(
        positions_object,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *active = (PyArrayObject *)PyArray_FROM_OTF(
        active_object,
        NPY_INT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (positions == NULL || active == NULL) {
        Py_XDECREF(positions);
        Py_XDECREF(active);
        return NULL;
    }
    if (
        PyArray_NDIM(positions) != 2
        || PyArray_DIM(positions, 1) != 3
        || PyArray_NDIM(active) != 1
    ) {
        Py_DECREF(positions);
        Py_DECREF(active);
        PyErr_SetString(PyExc_ValueError, "positions must be (N, 3) and active_indices must be 1D");
        return NULL;
    }

    npy_intp atom_count = PyArray_DIM(positions, 0);
    npy_intp active_count = PyArray_DIM(active, 0);
    if (active_count < 2) {
        Py_DECREF(positions);
        Py_DECREF(active);
        return empty_candidates();
    }

    double cell[3][3] = {{0.0}};
    double inverse[3][3] = {{0.0}};
    int periodic = cell_object != Py_None;
    PyArrayObject *cell_array = NULL;
    if (periodic) {
        cell_array = (PyArrayObject *)PyArray_FROM_OTF(
            cell_object,
            NPY_FLOAT64,
            NPY_ARRAY_IN_ARRAY
        );
        if (cell_array == NULL) {
            Py_DECREF(positions);
            Py_DECREF(active);
            return NULL;
        }
        if (
            PyArray_NDIM(cell_array) != 2
            || PyArray_DIM(cell_array, 0) != 3
            || PyArray_DIM(cell_array, 1) != 3
        ) {
            Py_DECREF(cell_array);
            Py_DECREF(positions);
            Py_DECREF(active);
            PyErr_SetString(PyExc_ValueError, "cell must have shape (3, 3)");
            return NULL;
        }
        double *cell_data = (double *)PyArray_DATA(cell_array);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                cell[row][column] = cell_data[row * 3 + column];
            }
        }
        if (!inverse3x3(cell, inverse)) {
            Py_DECREF(cell_array);
            Py_DECREF(positions);
            Py_DECREF(active);
            PyErr_SetString(PyExc_ValueError, "cell must be finite and invertible");
            return NULL;
        }
    }

    CellEntry *entries = (CellEntry *)malloc((size_t)active_count * sizeof(CellEntry));
    CellGroup *groups = (CellGroup *)malloc((size_t)active_count * sizeof(CellGroup));
    if (entries == NULL || groups == NULL) {
        free(entries);
        free(groups);
        Py_XDECREF(cell_array);
        Py_DECREF(positions);
        Py_DECREF(active);
        return PyErr_NoMemory();
    }

    float *position_data = (float *)PyArray_DATA(positions);
    int32_t *active_data = (int32_t *)PyArray_DATA(active);
    double origin[3] = {0.0, 0.0, 0.0};
    int64_t bin_count[3] = {1, 1, 1};
    int invalid_input = 0;
    CandidateBuffer candidates = {0};

    Py_BEGIN_ALLOW_THREADS
    if (!periodic) {
        int32_t first_atom = active_data[0];
        if (first_atom < 0 || first_atom >= atom_count) {
            invalid_input = 1;
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                origin[axis] = position_data[(size_t)first_atom * 3 + axis];
            }
            for (npy_intp i = 0; i < active_count; ++i) {
                int32_t atom = active_data[i];
                if (atom < 0 || atom >= atom_count) {
                    invalid_input = 1;
                    break;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    double value = position_data[(size_t)atom * 3 + axis];
                    if (!isfinite(value)) {
                        invalid_input = 1;
                        break;
                    }
                    if (value < origin[axis]) origin[axis] = value;
                }
                if (invalid_input) break;
            }
        }
    } else {
        for (int axis = 0; axis < 3; ++axis) {
            double reciprocal_norm = sqrt(
                inverse[0][axis] * inverse[0][axis]
                + inverse[1][axis] * inverse[1][axis]
                + inverse[2][axis] * inverse[2][axis]
            );
            double height = 1.0 / reciprocal_norm;
            double bins = floor(height / cutoff);
            bin_count[axis] = bins >= 1.0 && bins < (double)INT64_MAX ? (int64_t)bins : 1;
        }
    }

    if (!invalid_input) {
        for (npy_intp i = 0; i < active_count; ++i) {
            int32_t atom = active_data[i];
            if (atom < 0 || atom >= atom_count) {
                invalid_input = 1;
                break;
            }
            entries[i].atom = atom;
            if (periodic) {
                for (int axis = 0; axis < 3; ++axis) {
                    double fractional =
                        position_data[(size_t)atom * 3] * inverse[0][axis]
                        + position_data[(size_t)atom * 3 + 1] * inverse[1][axis]
                        + position_data[(size_t)atom * 3 + 2] * inverse[2][axis];
                    if (!isfinite(fractional)) {
                        invalid_input = 1;
                        break;
                    }
                    fractional -= floor(fractional);
                    int64_t coordinate = (int64_t)floor(fractional * bin_count[axis]);
                    if (coordinate >= bin_count[axis]) coordinate = bin_count[axis] - 1;
                    if (axis == 0) entries[i].x = coordinate;
                    else if (axis == 1) entries[i].y = coordinate;
                    else entries[i].z = coordinate;
                }
            } else {
                entries[i].x = (int64_t)floor(
                    (position_data[(size_t)atom * 3] - origin[0]) / cutoff
                );
                entries[i].y = (int64_t)floor(
                    (position_data[(size_t)atom * 3 + 1] - origin[1]) / cutoff
                );
                entries[i].z = (int64_t)floor(
                    (position_data[(size_t)atom * 3 + 2] - origin[2]) / cutoff
                );
            }
            if (invalid_input) break;
        }
    }

    npy_intp group_count = 0;
    if (!invalid_input) {
        qsort(entries, (size_t)active_count, sizeof(CellEntry), compare_entries);
        npy_intp start = 0;
        while (start < active_count) {
            npy_intp end = start + 1;
            while (
                end < active_count
                && entries[end].x == entries[start].x
                && entries[end].y == entries[start].y
                && entries[end].z == entries[start].z
            ) {
                ++end;
            }
            groups[group_count].x = entries[start].x;
            groups[group_count].y = entries[start].y;
            groups[group_count].z = entries[start].z;
            groups[group_count].start = start;
            groups[group_count].end = end;
            ++group_count;
            start = end;
        }
    }

    double cutoff2 = cutoff * cutoff;
    if (!invalid_input) {
        for (npy_intp group_index = 0; group_index < group_count && !candidates.failed; ++group_index) {
            npy_intp neighbors[27];
            int neighbor_count = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        int64_t x = groups[group_index].x + dx;
                        int64_t y = groups[group_index].y + dy;
                        int64_t z = groups[group_index].z + dz;
                        if (periodic) {
                            x = wrap_cell(x, bin_count[0]);
                            y = wrap_cell(y, bin_count[1]);
                            z = wrap_cell(z, bin_count[2]);
                        }
                        npy_intp neighbor = find_group(groups, group_count, x, y, z);
                        if (neighbor < group_index) continue;
                        int duplicate = 0;
                        for (int existing = 0; existing < neighbor_count; ++existing) {
                            if (neighbors[existing] == neighbor) {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (neighbor >= 0 && !duplicate) neighbors[neighbor_count++] = neighbor;
                    }
                }
            }

            for (int neighbor_i = 0; neighbor_i < neighbor_count; ++neighbor_i) {
                npy_intp neighbor = neighbors[neighbor_i];
                for (npy_intp left_i = groups[group_index].start; left_i < groups[group_index].end; ++left_i) {
                    npy_intp right_start = groups[neighbor].start;
                    if (neighbor == group_index) right_start = left_i + 1;
                    for (npy_intp right_i = right_start; right_i < groups[neighbor].end; ++right_i) {
                        int32_t left = entries[left_i].atom;
                        int32_t right = entries[right_i].atom;
                        float distance2 = pair_distance2(
                            position_data,
                            left,
                            right,
                            periodic,
                            cell,
                            inverse
                        );
                        if ((double)distance2 <= cutoff2) {
                            append_candidate(&candidates, distance2, left, right);
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(entries);
    free(groups);
    Py_XDECREF(cell_array);
    Py_DECREF(positions);
    Py_DECREF(active);
    if (invalid_input) {
        free(candidates.distance2);
        free(candidates.left);
        free(candidates.right);
        PyErr_SetString(PyExc_ValueError, "positions and active indices must be finite and in range");
        return NULL;
    }
    if (candidates.failed) {
        free(candidates.distance2);
        free(candidates.left);
        free(candidates.right);
        return PyErr_NoMemory();
    }

    npy_intp output_dims[1] = {(npy_intp)candidates.count};
    PyArrayObject *distance2_array = (PyArrayObject *)PyArray_SimpleNew(
        1,
        output_dims,
        NPY_FLOAT32
    );
    PyArrayObject *left_array = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_INT32);
    PyArrayObject *right_array = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_INT32);
    if (distance2_array == NULL || left_array == NULL || right_array == NULL) {
        Py_XDECREF(distance2_array);
        Py_XDECREF(left_array);
        Py_XDECREF(right_array);
        free(candidates.distance2);
        free(candidates.left);
        free(candidates.right);
        return NULL;
    }
    if (candidates.count > 0) {
        memcpy(
            PyArray_DATA(distance2_array),
            candidates.distance2,
            candidates.count * sizeof(float)
        );
        memcpy(PyArray_DATA(left_array), candidates.left, candidates.count * sizeof(int32_t));
        memcpy(PyArray_DATA(right_array), candidates.right, candidates.count * sizeof(int32_t));
    }
    free(candidates.distance2);
    free(candidates.left);
    free(candidates.right);
    return Py_BuildValue("NNN", distance2_array, left_array, right_array);
}


static PyMethodDef trajcore_methods[] = {
    {
        "connected_components",
        trajcore_connected_components,
        METH_VARARGS,
        "Return contiguous component labels and component sizes."
    },
    {
        "candidate_pairs",
        trajcore_candidate_pairs,
        METH_VARARGS,
        "Return neighbor pairs within a cutoff using a native cell list."
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef trajcore_module = {
    PyModuleDef_HEAD_INIT,
    "_trajcore",
    "Native trajectory topology hot paths.",
    -1,
    trajcore_methods,
};


PyMODINIT_FUNC PyInit__trajcore(void) {
    import_array();
    return PyModule_Create(&trajcore_module);
}
