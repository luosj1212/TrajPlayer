#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <float.h>
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


typedef struct {
    float distance2;
    int32_t left;
    int32_t right;
    npy_intp ordinal;
} BondCandidate;


typedef enum {
    XYZ_PARSE_OK = 0,
    XYZ_PARSE_TRUNCATED,
    XYZ_PARSE_MISSING_COLUMNS,
    XYZ_PARSE_INVALID_POSITION,
    XYZ_PARSE_INVALID_IDENTITY,
    XYZ_PARSE_IDENTITY_MISMATCH
} XyzParseStatus;


typedef struct {
    XyzParseStatus status;
    npy_intp atom;
    int column;
} XyzParseError;


static const char *CHEMICAL_SYMBOLS[] = {
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};


static int ascii_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\r'
        || value == '\n' || value == '\v' || value == '\f';
}


static int ascii_alpha(unsigned char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}


static unsigned char ascii_upper(unsigned char value) {
    return value >= 'a' && value <= 'z' ? (unsigned char)(value - ('a' - 'A')) : value;
}


static unsigned char ascii_lower(unsigned char value) {
    return value >= 'A' && value <= 'Z' ? (unsigned char)(value + ('a' - 'A')) : value;
}


static int token_equals_ascii_casefold(
    const unsigned char *start,
    const unsigned char *end,
    const char *literal
) {
    const unsigned char *cursor = start;
    const unsigned char *expected = (const unsigned char *)literal;
    while (cursor < end && *expected != '\0') {
        if (ascii_lower(*cursor) != ascii_lower(*expected)) return 0;
        ++cursor;
        ++expected;
    }
    return cursor == end && *expected == '\0';
}


static int parse_float_token(
    const unsigned char *start,
    const unsigned char *end,
    float *output
) {
    const unsigned char *cursor = start;
    int negative = 0;
    if (cursor < end && (*cursor == '+' || *cursor == '-')) {
        negative = *cursor == '-';
        ++cursor;
    }
    if (cursor >= end) return 0;

    if (token_equals_ascii_casefold(cursor, end, "nan")) {
        *output = negative ? -NAN : NAN;
        return 1;
    }
    if (
        token_equals_ascii_casefold(cursor, end, "inf")
        || token_equals_ascii_casefold(cursor, end, "infinity")
    ) {
        *output = negative ? -INFINITY : INFINITY;
        return 1;
    }

    double value = 0.0;
    int digits = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        value = value * 10.0 + (double)(*cursor - '0');
        ++cursor;
        ++digits;
    }
    if (cursor < end && *cursor == '.') {
        ++cursor;
        double scale = 0.1;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            value += (double)(*cursor - '0') * scale;
            scale *= 0.1;
            ++cursor;
            ++digits;
        }
    }
    if (digits == 0) return 0;

    int exponent = 0;
    int exponent_negative = 0;
    if (cursor < end && (*cursor == 'e' || *cursor == 'E')) {
        ++cursor;
        if (cursor < end && (*cursor == '+' || *cursor == '-')) {
            exponent_negative = *cursor == '-';
            ++cursor;
        }
        int exponent_digits = 0;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            if (exponent < 100000) exponent = exponent * 10 + (*cursor - '0');
            ++cursor;
            ++exponent_digits;
        }
        if (exponent_digits == 0) return 0;
    }
    if (cursor != end) return 0;
    if (exponent != 0) {
        value *= pow(10.0, exponent_negative ? -(double)exponent : (double)exponent);
    }
    *output = (float)(negative ? -value : value);
    return 1;
}


static int parse_atomic_number_token(
    const unsigned char *start,
    const unsigned char *end,
    uint16_t *output
) {
    const unsigned char *cursor = start;
    int negative = 0;
    if (cursor < end && (*cursor == '+' || *cursor == '-')) {
        negative = *cursor == '-';
        ++cursor;
    }
    if (negative || cursor >= end) return 0;
    uint32_t value = 0;
    int digits = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        uint32_t digit = (uint32_t)(*cursor - '0');
        if (value > (UINT16_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
        ++cursor;
        ++digits;
    }
    if (digits == 0 || cursor != end) return 0;
    *output = (uint16_t)value;
    return 1;
}


static uint16_t symbol_token_to_atomic_number(
    const unsigned char *start,
    const unsigned char *end
) {
    unsigned char letters[2] = {0, 0};
    int letter_count = 0;
    const unsigned char *cursor = start;
    while (cursor < end && !ascii_alpha(*cursor)) ++cursor;
    while (cursor < end && ascii_alpha(*cursor) && letter_count < 2) {
        letters[letter_count++] = *cursor;
        ++cursor;
    }
    if (letter_count == 0) return 0;

    char candidate[3] = {(char)ascii_upper(letters[0]), '\0', '\0'};
    if (letter_count >= 2) {
        candidate[1] = (char)ascii_lower(letters[1]);
        for (uint16_t number = 1; number <= 118; ++number) {
            if (strcmp(candidate, CHEMICAL_SYMBOLS[number]) == 0) return number;
        }
        candidate[1] = '\0';
    }
    for (uint16_t number = 1; number <= 118; ++number) {
        if (strcmp(candidate, CHEMICAL_SYMBOLS[number]) == 0) return number;
    }
    return 0;
}


static int compare_entries(const void *lhs, const void *rhs) {
    const CellEntry *a = (const CellEntry *)lhs;
    const CellEntry *b = (const CellEntry *)rhs;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    if (a->z != b->z) return a->z < b->z ? -1 : 1;
    if (a->atom != b->atom) return a->atom < b->atom ? -1 : 1;
    return 0;
}


static int compare_bond_candidates(const void *lhs, const void *rhs) {
    const BondCandidate *a = (const BondCandidate *)lhs;
    const BondCandidate *b = (const BondCandidate *)rhs;
    if (a->distance2 != b->distance2) return a->distance2 < b->distance2 ? -1 : 1;
    if (a->left != b->left) return a->left < b->left ? -1 : 1;
    if (a->right != b->right) return a->right < b->right ? -1 : 1;
    if (a->ordinal != b->ordinal) return a->ordinal < b->ordinal ? -1 : 1;
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


static PyObject *trajcore_coarse_depth_order(PyObject *self, PyObject *args) {
    PyObject *depth_object;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &depth_object)) return NULL;

    PyArrayObject *depth = (PyArrayObject *)PyArray_FROM_OTF(
        depth_object,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (depth == NULL) return NULL;
    if (PyArray_NDIM(depth) != 1) {
        Py_DECREF(depth);
        PyErr_SetString(PyExc_ValueError, "view_depth must be one-dimensional");
        return NULL;
    }

    npy_intp count = PyArray_DIM(depth, 0);
    npy_intp output_dims[1] = {count};
    PyArrayObject *order = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_INT64);
    if (order == NULL) {
        Py_DECREF(depth);
        return NULL;
    }

    float *values = (float *)PyArray_DATA(depth);
    int64_t *output = (int64_t *)PyArray_DATA(order);
    if (count <= 1) {
        if (count == 1) output[0] = 0;
        Py_DECREF(depth);
        return (PyObject *)order;
    }

    npy_intp bin_counts[256] = {0};
    npy_intp bin_offsets[256] = {0};
    float minimum = values[0];
    float maximum = values[0];
    int valid_span = 1;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp index = 0; index < count; ++index) {
        float value = values[index];
        if (!isfinite(value)) {
            valid_span = 0;
            break;
        }
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
    }

    double span = (double)maximum - (double)minimum;
    if (!isfinite(span) || span <= (double)FLT_EPSILON) valid_span = 0;

    if (valid_span) {
        float scale = (float)(255.0 / span);
        for (npy_intp index = 0; index < count; ++index) {
            float scaled = (values[index] - minimum) * scale;
            int bin = (int)scaled;
            if (bin < 0) bin = 0;
            else if (bin > 255) bin = 255;
            bin_counts[bin] += 1;
        }

        npy_intp offset = 0;
        for (int bin = 255; bin >= 0; --bin) {
            bin_offsets[bin] = offset;
            offset += bin_counts[bin];
        }

        for (npy_intp index = count; index-- > 0;) {
            float scaled = (values[index] - minimum) * scale;
            int bin = (int)scaled;
            if (bin < 0) bin = 0;
            else if (bin > 255) bin = 255;
            output[bin_offsets[bin]++] = (int64_t)index;
        }
    } else {
        for (npy_intp index = 0; index < count; ++index) {
            output[index] = (int64_t)(count - 1 - index);
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(depth);
    return (PyObject *)order;
}


static float projected_depth(
    const float *positions,
    npy_intp index,
    const float forward[3]
) {
    const float *position = positions + (size_t)index * 3;
    return position[0] * forward[0]
        + position[1] * forward[1]
        + position[2] * forward[2];
}


static PyObject *trajcore_coarse_position_depth_order(PyObject *self, PyObject *args) {
    PyObject *positions_object;
    PyObject *forward_object;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &positions_object, &forward_object)) return NULL;

    PyArrayObject *positions = (PyArrayObject *)PyArray_FROM_OTF(
        positions_object,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (positions == NULL) return NULL;
    PyArrayObject *forward = (PyArrayObject *)PyArray_FROM_OTF(
        forward_object,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (forward == NULL) {
        Py_DECREF(positions);
        return NULL;
    }
    if (PyArray_NDIM(positions) != 2 || PyArray_DIM(positions, 1) != 3) {
        Py_DECREF(positions);
        Py_DECREF(forward);
        PyErr_SetString(PyExc_ValueError, "positions must have shape (N, 3)");
        return NULL;
    }
    if (PyArray_NDIM(forward) != 1 || PyArray_DIM(forward, 0) != 3) {
        Py_DECREF(positions);
        Py_DECREF(forward);
        PyErr_SetString(PyExc_ValueError, "camera_forward must have shape (3,)");
        return NULL;
    }

    npy_intp count = PyArray_DIM(positions, 0);
    npy_intp output_dims[1] = {count};
    PyArrayObject *order = (PyArrayObject *)PyArray_SimpleNew(1, output_dims, NPY_INT64);
    if (order == NULL) {
        Py_DECREF(positions);
        Py_DECREF(forward);
        return NULL;
    }
    const float *position_data = (const float *)PyArray_DATA(positions);
    const float *forward_data = (const float *)PyArray_DATA(forward);
    int64_t *output = (int64_t *)PyArray_DATA(order);
    if (count <= 1) {
        if (count == 1) output[0] = 0;
        Py_DECREF(positions);
        Py_DECREF(forward);
        return (PyObject *)order;
    }

    npy_intp bin_counts[256] = {0};
    npy_intp bin_offsets[256] = {0};
    float minimum = 0.0f;
    float maximum = 0.0f;
    int valid_span = 1;

    Py_BEGIN_ALLOW_THREADS
    minimum = projected_depth(position_data, 0, forward_data);
    maximum = minimum;
    if (!isfinite(minimum)) valid_span = 0;
    for (npy_intp index = 1; index < count && valid_span; ++index) {
        float value = projected_depth(position_data, index, forward_data);
        if (!isfinite(value)) {
            valid_span = 0;
            break;
        }
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
    }

    double span = (double)maximum - (double)minimum;
    if (!isfinite(span) || span <= (double)FLT_EPSILON) valid_span = 0;
    if (valid_span) {
        float scale = (float)(255.0 / span);
        for (npy_intp index = 0; index < count; ++index) {
            float scaled = (projected_depth(position_data, index, forward_data) - minimum) * scale;
            int bin = (int)scaled;
            if (bin < 0) bin = 0;
            else if (bin > 255) bin = 255;
            bin_counts[bin] += 1;
        }
        npy_intp offset = 0;
        for (int bin = 255; bin >= 0; --bin) {
            bin_offsets[bin] = offset;
            offset += bin_counts[bin];
        }
        for (npy_intp index = count; index-- > 0;) {
            float scaled = (projected_depth(position_data, index, forward_data) - minimum) * scale;
            int bin = (int)scaled;
            if (bin < 0) bin = 0;
            else if (bin > 255) bin = 255;
            output[bin_offsets[bin]++] = (int64_t)index;
        }
    } else {
        for (npy_intp index = 0; index < count; ++index) {
            output[index] = (int64_t)(count - 1 - index);
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(positions);
    Py_DECREF(forward);
    return (PyObject *)order;
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


static PyObject *trajcore_select_valence_bonds(PyObject *self, PyObject *args) {
    PyObject *distance2_object;
    PyObject *left_object;
    PyObject *right_object;
    PyObject *caps_object;
    (void)self;
    if (!PyArg_ParseTuple(
        args,
        "OOOO",
        &distance2_object,
        &left_object,
        &right_object,
        &caps_object
    )) {
        return NULL;
    }

    PyArrayObject *distance2 = (PyArrayObject *)PyArray_FROM_OTF(
        distance2_object,
        NPY_FLOAT32,
        NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *left = (PyArrayObject *)PyArray_FROM_OTF(
        left_object,
        NPY_INT32,
        NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *right = (PyArrayObject *)PyArray_FROM_OTF(
        right_object,
        NPY_INT32,
        NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *caps = (PyArrayObject *)PyArray_FROM_OTF(
        caps_object,
        NPY_UINT8,
        NPY_ARRAY_IN_ARRAY
    );
    if (distance2 == NULL || left == NULL || right == NULL || caps == NULL) {
        Py_XDECREF(distance2);
        Py_XDECREF(left);
        Py_XDECREF(right);
        Py_XDECREF(caps);
        return NULL;
    }

    int shapes_valid =
        PyArray_NDIM(distance2) == 1
        && PyArray_NDIM(left) == 1
        && PyArray_NDIM(right) == 1
        && PyArray_NDIM(caps) == 1
        && PyArray_DIM(distance2, 0) == PyArray_DIM(left, 0)
        && PyArray_DIM(distance2, 0) == PyArray_DIM(right, 0);
    if (!shapes_valid) {
        Py_DECREF(distance2);
        Py_DECREF(left);
        Py_DECREF(right);
        Py_DECREF(caps);
        PyErr_SetString(
            PyExc_ValueError,
            "distance2, left, right, and caps must be one-dimensional; candidate lengths must match"
        );
        return NULL;
    }

    npy_intp candidate_count = PyArray_DIM(distance2, 0);
    npy_intp atom_count = PyArray_DIM(caps, 0);
    if (atom_count > INT32_MAX) {
        Py_DECREF(distance2);
        Py_DECREF(left);
        Py_DECREF(right);
        Py_DECREF(caps);
        PyErr_SetString(PyExc_ValueError, "caps length must fit in int32");
        return NULL;
    }

    float *distance_data = (float *)PyArray_DATA(distance2);
    int32_t *left_data = (int32_t *)PyArray_DATA(left);
    int32_t *right_data = (int32_t *)PyArray_DATA(right);
    uint8_t *cap_data = (uint8_t *)PyArray_DATA(caps);
    BondCandidate *candidates = NULL;
    uint16_t *degrees = NULL;
    int32_t *bond_data = NULL;
    if (candidate_count > 0) {
        candidates = (BondCandidate *)malloc(
            (size_t)candidate_count * sizeof(BondCandidate)
        );
    }
    if (atom_count > 0) {
        degrees = (uint16_t *)calloc((size_t)atom_count, sizeof(uint16_t));
    }
    if ((candidate_count > 0 && candidates == NULL) || (atom_count > 0 && degrees == NULL)) {
        free(candidates);
        free(degrees);
        Py_DECREF(distance2);
        Py_DECREF(left);
        Py_DECREF(right);
        Py_DECREF(caps);
        return PyErr_NoMemory();
    }

    uint64_t cap_sum = 0;
    for (npy_intp atom = 0; atom < atom_count; ++atom) cap_sum += cap_data[atom];
    npy_intp max_bond_count = candidate_count;
    if ((uint64_t)max_bond_count > cap_sum / 2u) {
        max_bond_count = (npy_intp)(cap_sum / 2u);
    }
    if (max_bond_count > 0) {
        bond_data = (int32_t *)malloc((size_t)max_bond_count * 2u * sizeof(int32_t));
        if (bond_data == NULL) {
            free(candidates);
            free(degrees);
            Py_DECREF(distance2);
            Py_DECREF(left);
            Py_DECREF(right);
            Py_DECREF(caps);
            return PyErr_NoMemory();
        }
    }

    int invalid_input = 0;
    npy_intp bond_count = 0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp candidate = 0; candidate < candidate_count; ++candidate) {
        int32_t i = left_data[candidate];
        int32_t j = right_data[candidate];
        float value = distance_data[candidate];
        if (
            !isfinite(value)
            || i < 0
            || i >= atom_count
            || j < 0
            || j >= atom_count
            || i == j
        ) {
            invalid_input = 1;
            break;
        }
        candidates[candidate].distance2 = value;
        candidates[candidate].left = i;
        candidates[candidate].right = j;
        candidates[candidate].ordinal = candidate;
    }
    if (!invalid_input && candidate_count > 1) {
        qsort(
            candidates,
            (size_t)candidate_count,
            sizeof(BondCandidate),
            compare_bond_candidates
        );
    }
    if (!invalid_input) {
        for (npy_intp candidate = 0; candidate < candidate_count; ++candidate) {
            int32_t i = candidates[candidate].left;
            int32_t j = candidates[candidate].right;
            if (degrees[i] >= cap_data[i] || degrees[j] >= cap_data[j]) continue;
            degrees[i] += 1;
            degrees[j] += 1;
            bond_data[bond_count * 2] = i;
            bond_data[bond_count * 2 + 1] = j;
            ++bond_count;
            if (bond_count == max_bond_count) break;
        }
    }
    Py_END_ALLOW_THREADS

    free(candidates);
    free(degrees);
    Py_DECREF(distance2);
    Py_DECREF(left);
    Py_DECREF(right);
    Py_DECREF(caps);
    if (invalid_input) {
        free(bond_data);
        PyErr_SetString(
            PyExc_ValueError,
            "candidate distances must be finite and atom indices must be distinct and in range"
        );
        return NULL;
    }

    npy_intp output_dims[2] = {bond_count, 2};
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(2, output_dims, NPY_INT32);
    if (output == NULL) {
        free(bond_data);
        return NULL;
    }
    if (bond_count > 0) {
        memcpy(PyArray_DATA(output), bond_data, (size_t)bond_count * 2u * sizeof(int32_t));
    }
    free(bond_data);
    return (PyObject *)output;
}


static PyObject *trajcore_xyz_read_frame_into(PyObject *self, PyObject *args) {
    PyObject *source_object;
    PyObject *positions_object;
    PyObject *layout_object;
    PyObject *expected_object;
    Py_ssize_t data_start;
    Py_ssize_t data_end;
    int identity_is_atomic_number;
    (void)self;
    if (!PyArg_ParseTuple(
        args,
        "OnnOOOp",
        &source_object,
        &data_start,
        &data_end,
        &positions_object,
        &layout_object,
        &expected_object,
        &identity_is_atomic_number
    )) {
        return NULL;
    }

    if (!PyArray_Check(positions_object)) {
        PyErr_SetString(PyExc_TypeError, "positions must be a NumPy array");
        return NULL;
    }
    PyArrayObject *positions = (PyArrayObject *)positions_object;
    if (
        PyArray_TYPE(positions) != NPY_FLOAT32
        || PyArray_NDIM(positions) != 2
        || PyArray_DIM(positions, 1) != 3
        || !PyArray_IS_C_CONTIGUOUS(positions)
        || !PyArray_ISWRITEABLE(positions)
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "positions must be a writable C-contiguous float32 array with shape (N, 3)"
        );
        return NULL;
    }

    PyArrayObject *layout = (PyArrayObject *)PyArray_FROM_OTF(
        layout_object,
        NPY_INT32,
        NPY_ARRAY_IN_ARRAY
    );
    if (layout == NULL) return NULL;
    PyArrayObject *expected = (PyArrayObject *)PyArray_FROM_OTF(
        expected_object,
        NPY_UINT16,
        NPY_ARRAY_IN_ARRAY
    );
    if (expected == NULL) {
        Py_DECREF(layout);
        return NULL;
    }
    npy_intp atom_count = PyArray_DIM(positions, 0);
    if (PyArray_NDIM(layout) != 1 || PyArray_DIM(layout, 0) != 5) {
        Py_DECREF(layout);
        Py_DECREF(expected);
        PyErr_SetString(PyExc_ValueError, "layout must contain five int32 values");
        return NULL;
    }
    if (PyArray_NDIM(expected) != 1 || PyArray_DIM(expected, 0) != atom_count) {
        Py_DECREF(layout);
        Py_DECREF(expected);
        PyErr_SetString(PyExc_ValueError, "expected_atom_numbers must have shape (N,)");
        return NULL;
    }

    int32_t *layout_data = (int32_t *)PyArray_DATA(layout);
    int identity_column = layout_data[0];
    int position_columns[3] = {layout_data[1], layout_data[2], layout_data[3]};
    int expected_columns = layout_data[4];
    if (
        expected_columns <= 0
        || identity_column < 0
        || identity_column >= expected_columns
        || position_columns[0] < 0
        || position_columns[0] >= expected_columns
        || position_columns[1] < 0
        || position_columns[1] >= expected_columns
        || position_columns[2] < 0
        || position_columns[2] >= expected_columns
    ) {
        Py_DECREF(layout);
        Py_DECREF(expected);
        PyErr_SetString(PyExc_ValueError, "XYZ column layout is invalid");
        return NULL;
    }

    Py_buffer source = {0};
    if (PyObject_GetBuffer(source_object, &source, PyBUF_CONTIG_RO) < 0) {
        Py_DECREF(layout);
        Py_DECREF(expected);
        return NULL;
    }
    if (
        data_start < 0
        || data_end < data_start
        || data_end > source.len
    ) {
        PyBuffer_Release(&source);
        Py_DECREF(layout);
        Py_DECREF(expected);
        PyErr_SetString(PyExc_ValueError, "XYZ byte range lies outside the source buffer");
        return NULL;
    }

    const unsigned char *cursor = (const unsigned char *)source.buf + data_start;
    const unsigned char *end = (const unsigned char *)source.buf + data_end;
    float *position_data = (float *)PyArray_DATA(positions);
    uint16_t *expected_numbers = (uint16_t *)PyArray_DATA(expected);
    XyzParseError error = {XYZ_PARSE_OK, -1, -1};

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp atom = 0; atom < atom_count; ++atom) {
        if (cursor >= end) {
            error.status = XYZ_PARSE_TRUNCATED;
            error.atom = atom;
            break;
        }
        const unsigned char *line_end = cursor;
        while (line_end < end && *line_end != '\n') ++line_end;
        const unsigned char *token_cursor = cursor;

        for (int column = 0; column < expected_columns; ++column) {
            while (token_cursor < line_end && ascii_space(*token_cursor)) ++token_cursor;
            if (token_cursor >= line_end) {
                error.status = XYZ_PARSE_MISSING_COLUMNS;
                error.atom = atom;
                error.column = column;
                break;
            }
            const unsigned char *token_start = token_cursor;
            while (token_cursor < line_end && !ascii_space(*token_cursor)) ++token_cursor;
            const unsigned char *token_end = token_cursor;

            if (column == identity_column) {
                uint16_t actual_number = 0;
                int valid_identity = identity_is_atomic_number
                    ? parse_atomic_number_token(token_start, token_end, &actual_number)
                    : 1;
                if (!identity_is_atomic_number) {
                    actual_number = symbol_token_to_atomic_number(token_start, token_end);
                }
                if (!valid_identity) {
                    error.status = XYZ_PARSE_INVALID_IDENTITY;
                    error.atom = atom;
                    error.column = column;
                    break;
                }
                if (actual_number != expected_numbers[atom]) {
                    error.status = XYZ_PARSE_IDENTITY_MISMATCH;
                    error.atom = atom;
                    error.column = column;
                    break;
                }
            }

            for (int axis = 0; axis < 3; ++axis) {
                if (column != position_columns[axis]) continue;
                float value;
                if (!parse_float_token(token_start, token_end, &value)) {
                    error.status = XYZ_PARSE_INVALID_POSITION;
                    error.atom = atom;
                    error.column = column;
                    break;
                }
                position_data[(size_t)atom * 3 + axis] = value;
            }
            if (error.status != XYZ_PARSE_OK) break;
        }
        if (error.status != XYZ_PARSE_OK) break;
        cursor = line_end < end ? line_end + 1 : line_end;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&source);
    Py_DECREF(layout);
    Py_DECREF(expected);
    if (error.status != XYZ_PARSE_OK) {
        const char *reason = "could not be parsed";
        if (error.status == XYZ_PARSE_TRUNCATED) reason = "frame ended early";
        else if (error.status == XYZ_PARSE_MISSING_COLUMNS) reason = "atom row has too few columns";
        else if (error.status == XYZ_PARSE_INVALID_POSITION) reason = "position is not numeric";
        else if (error.status == XYZ_PARSE_INVALID_IDENTITY) reason = "atomic identity is invalid";
        else if (error.status == XYZ_PARSE_IDENTITY_MISMATCH) reason = "atomic identity differs from frame 0";
        PyErr_Format(
            PyExc_ValueError,
            "XYZ native parser failed at atom %zd, column %d: %s",
            error.atom,
            error.column,
            reason
        );
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef trajcore_methods[] = {
    {
        "xyz_read_frame_into",
        trajcore_xyz_read_frame_into,
        METH_VARARGS,
        "Parse XYZ atom rows directly into a caller-owned float32 array."
    },
    {
        "coarse_position_depth_order",
        trajcore_coarse_position_depth_order,
        METH_VARARGS,
        "Project canonical positions and return a far-to-near 256-bin order."
    },
    {
        "coarse_depth_order",
        trajcore_coarse_depth_order,
        METH_VARARGS,
        "Return a far-to-near stable reverse order using 256 counting bins."
    },
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
    {
        "select_valence_bonds",
        trajcore_select_valence_bonds,
        METH_VARARGS,
        "Sort bond candidates and apply per-atom valence caps."
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
