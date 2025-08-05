#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// Define the matrix structure
typedef struct {
    int rows;
    int cols;
    int byte_width;
    uint8_t** data;
} BitMatrix;

// Allocate a bit-packed binary matrix
BitMatrix* create_matrix(int rows, int cols) {
    BitMatrix* m = malloc(sizeof(BitMatrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->byte_width = (cols + 7) / 8;

    m->data = malloc(rows * sizeof(uint8_t*));
    if (!m->data) {
        free(m);
        return NULL;
    }

    for (int i = 0; i < rows; ++i) {
        m->data[i] = calloc(m->byte_width, sizeof(uint8_t));
        if (!m->data[i]) {
            for (int j = 0; j < i; ++j) free(m->data[j]);
            free(m->data);
            free(m);
            return NULL;
        }
    }

    return m;
}

// Free the matrix memory
void free_matrix(BitMatrix* m) {
    if (!m) return;
    for (int i = 0; i < m->rows; ++i) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Set a bit to 1
void set_bit(BitMatrix* m, int i, int j) {
    m->data[i][j / 8] |= (1 << (7 - (j % 8)));
}

// Clear a bit to 0
void clear_bit(BitMatrix* m, int i, int j) {
    m->data[i][j / 8] &= ~(1 << (7 - (j % 8)));
}

// Get a bit value (0 or 1)
int get_bit(BitMatrix* m, int i, int j) {
    return (m->data[i][j / 8] >> (7 - (j % 8))) & 1;
}

// Print the entire matrix to stdout
void print_matrix(BitMatrix* m) {
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            printf("%d ", get_bit(m, i, j));
        }
        printf("\n");
    }
}

// Set exactly `n` random bits to 1 and the rest to 0
void set_random_n_ones(BitMatrix* m, int n) {
    int total = m->rows * m->cols;
    if (n > total) return;

    // Clear all bits
    for (int i = 0; i < m->rows; ++i) {
        memset(m->data[i], 0, m->byte_width);
    }

    int* indices = malloc(sizeof(int) * total);
    if (!indices) return;

    for (int i = 0; i < total; ++i) {
        indices[i] = i;
    }

    srand((unsigned int)time(NULL));

    // Fisher-Yates shuffle
    for (int i = total - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    for (int k = 0; k < n; ++k) {
        int idx = indices[k];
        int row = idx / m->cols;
        int col = idx % m->cols;
        set_bit(m, row, col);
    }

    free(indices);
}

// Return raw data memory usage in bytes
int get_memory_usage(BitMatrix* m) {
    if (!m) return 0;
    return m->rows * m->byte_width;
}
