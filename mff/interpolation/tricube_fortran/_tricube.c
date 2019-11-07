#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*****************************************************************
 
    Based on eqtools package.
    https://github.com/PSFCPlasmaTools/eqtools

******************************************************************/

int A[64][64] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 10, -8, 2, 0, 0, 0, 0, 4, -10, 8, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -6, 6, -2, 0, 0, 0, 0, -2, 6, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, -20, 0, 0, 0, 16, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 4, 0, 10, 0, -10, 0, -8, 0, 8, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, -20, 16, -4, -20, 50, -40, 10, 16, -40, 32, -8, -4, 10, -8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 12, -12, 4, 10, -30, 30, -10, -8, 24, -24, 8, 2, -6, 6, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 12, 0, 0, 0, -12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, -6, 0, 6, 0, 6, 0, -6, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 10, -8, 2, 12, -30, 24, -6, -12, 30, -24, 6, 4, -10, 8, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -6, 6, -2, -6, 18, -18, 6, 6, -18, 18, -6, -2, 6, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, -4, 10, -8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -10, 8, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 2, -6, 6, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 6, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 2, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {-1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {2, -5, 4, -1, 0, 0, 0, 0, -2, 5, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 5, -4, 1, 0, 0, 0, 0, 2, -5, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {-1, 3, -3, 1, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0, 0, 0, -1, 3, -3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, -4, 0, 0, 0, 10, 0, 0, 0, -8, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, -10, 0, 0, 0, 8, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {2, 0, -2, 0, -5, 0, 5, 0, 4, 0, -4, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 5, 0, -5, 0, -4, 0, 4, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {-4, 10, -8, 2, 10, -25, 20, -5, -8, 20, -16, 4, 2, -5, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -10, 8, -2, -10, 25, -20, 5, 8, -20, 16, -4, -2, 5, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {2, -6, 6, -2, -5, 15, -15, 5, 4, -12, 12, -4, -1, 3, -3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 6, -6, 2, 5, -15, 15, -5, -4, 12, -12, 4, 1, -3, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 2, 0, 0, 0, -6, 0, 0, 0, 6, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 6, 0, 0, 0, -6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {-1, 0, 1, 0, 3, 0, -3, 0, -3, 0, 3, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, -3, 0, 3, 0, 3, 0, -3, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {2, -5, 4, -1, -6, 15, -12, 3, 6, -15, 12, -3, -2, 5, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 5, -4, 1, 6, -15, 12, -3, -6, 15, -12, 3, 2, -5, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {-1, 3, -3, 1, 3, -9, 9, -3, -3, 9, -9, 3, 1, -3, 3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -3, 3, -1, -3, 9, -9, 3, 3, -9, 9, -3, -1, 3, -3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -20, 50, -40, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, -40, 32, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 10, -8, 2, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, -30, 30, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8, 24, -24, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -6, 6, -2, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, -4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, -8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0},
				 {2, 0, -2, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, -5, 0, 5, 0, 0, 0, 0, 0, 5, 0, -5, 0, 0, 0, 0, 0, 4, 0, -4, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0},
				 {-4, 10, -8, 2, 0, 0, 0, 0, 4, -10, 8, -2, 0, 0, 0, 0, 10, -25, 20, -5, 0, 0, 0, 0, -10, 25, -20, 5, 0, 0, 0, 0, -8, 20, -16, 4, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0, 2, -5, 4, -1, 0, 0, 0, 0, -2, 5, -4, 1, 0, 0, 0, 0},
				 {2, -6, 6, -2, 0, 0, 0, 0, -2, 6, -6, 2, 0, 0, 0, 0, -5, 15, -15, 5, 0, 0, 0, 0, 5, -15, 15, -5, 0, 0, 0, 0, 4, -12, 12, -4, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0, -1, 3, -3, 1, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0, 0, 0},
				 {0, 8, 0, 0, 0, -20, 0, 0, 0, 16, 0, 0, 0, -4, 0, 0, 0, -20, 0, 0, 0, 50, 0, 0, 0, -40, 0, 0, 0, 10, 0, 0, 0, 16, 0, 0, 0, -40, 0, 0, 0, 32, 0, 0, 0, -8, 0, 0, 0, -4, 0, 0, 0, 10, 0, 0, 0, -8, 0, 0, 0, 2, 0, 0},
				 {-4, 0, 4, 0, 10, 0, -10, 0, -8, 0, 8, 0, 2, 0, -2, 0, 10, 0, -10, 0, -25, 0, 25, 0, 20, 0, -20, 0, -5, 0, 5, 0, -8, 0, 8, 0, 20, 0, -20, 0, -16, 0, 16, 0, 4, 0, -4, 0, 2, 0, -2, 0, -5, 0, 5, 0, 4, 0, -4, 0, -1, 0, 1, 0},
				 {8, -20, 16, -4, -20, 50, -40, 10, 16, -40, 32, -8, -4, 10, -8, 2, -20, 50, -40, 10, 50, -125, 100, -25, -40, 100, -80, 20, 10, -25, 20, -5, 16, -40, 32, -8, -40, 100, -80, 20, 32, -80, 64, -16, -8, 20, -16, 4, -4, 10, -8, 2, 10, -25, 20, -5, -8, 20, -16, 4, 2, -5, 4, -1},
				 {-4, 12, -12, 4, 10, -30, 30, -10, -8, 24, -24, 8, 2, -6, 6, -2, 10, -30, 30, -10, -25, 75, -75, 25, 20, -60, 60, -20, -5, 15, -15, 5, -8, 24, -24, 8, 20, -60, 60, -20, -16, 48, -48, 16, 4, -12, 12, -4, 2, -6, 6, -2, -5, 15, -15, 5, 4, -12, 12, -4, -1, 3, -3, 1},
				 {0, -4, 0, 0, 0, 12, 0, 0, 0, -12, 0, 0, 0, 4, 0, 0, 0, 10, 0, 0, 0, -30, 0, 0, 0, 30, 0, 0, 0, -10, 0, 0, 0, -8, 0, 0, 0, 24, 0, 0, 0, -24, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, -6, 0, 0, 0, 6, 0, 0, 0, -2, 0, 0},
				 {2, 0, -2, 0, -6, 0, 6, 0, 6, 0, -6, 0, -2, 0, 2, 0, -5, 0, 5, 0, 15, 0, -15, 0, -15, 0, 15, 0, 5, 0, -5, 0, 4, 0, -4, 0, -12, 0, 12, 0, 12, 0, -12, 0, -4, 0, 4, 0, -1, 0, 1, 0, 3, 0, -3, 0, -3, 0, 3, 0, 1, 0, -1, 0},
				 {-4, 10, -8, 2, 12, -30, 24, -6, -12, 30, -24, 6, 4, -10, 8, -2, 10, -25, 20, -5, -30, 75, -60, 15, 30, -75, 60, -15, -10, 25, -20, 5, -8, 20, -16, 4, 24, -60, 48, -12, -24, 60, -48, 12, 8, -20, 16, -4, 2, -5, 4, -1, -6, 15, -12, 3, 6, -15, 12, -3, -2, 5, -4, 1},
				 {2, -6, 6, -2, -6, 18, -18, 6, 6, -18, 18, -6, -2, 6, -6, 2, -5, 15, -15, 5, 15, -45, 45, -15, -15, 45, -45, 15, 5, -15, 15, -5, 4, -12, 12, -4, -12, 36, -36, 12, 12, -36, 36, -12, -4, 12, -12, 4, -1, 3, -3, 1, 3, -9, 9, -3, -3, 9, -9, 3, 1, -3, 3, -1},
				 {0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, -4, 10, -8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, -30, 24, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -12, 30, -24, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -10, 8, -2, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 0, 0, 0, 2, -6, 6, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 18, -18, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -18, 18, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 6, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0},
				 {0, 2, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0},
				 {-1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 3, 0, -3, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 3, 0, -3, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0},
				 {2, -5, 4, -1, 0, 0, 0, 0, -2, 5, -4, 1, 0, 0, 0, 0, -6, 15, -12, 3, 0, 0, 0, 0, 6, -15, 12, -3, 0, 0, 0, 0, 6, -15, 12, -3, 0, 0, 0, 0, -6, 15, -12, 3, 0, 0, 0, 0, -2, 5, -4, 1, 0, 0, 0, 0, 2, -5, 4, -1, 0, 0, 0, 0},
				 {-1, 3, -3, 1, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0, 0, 0, 3, -9, 9, -3, 0, 0, 0, 0, -3, 9, -9, 3, 0, 0, 0, 0, -3, 9, -9, 3, 0, 0, 0, 0, 3, -9, 9, -3, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0, 0, 0, -1, 3, -3, 1, 0, 0, 0, 0},
				 {0, -4, 0, 0, 0, 10, 0, 0, 0, -8, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, -30, 0, 0, 0, 24, 0, 0, 0, -6, 0, 0, 0, -12, 0, 0, 0, 30, 0, 0, 0, -24, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, -10, 0, 0, 0, 8, 0, 0, 0, -2, 0, 0},
				 {2, 0, -2, 0, -5, 0, 5, 0, 4, 0, -4, 0, -1, 0, 1, 0, -6, 0, 6, 0, 15, 0, -15, 0, -12, 0, 12, 0, 3, 0, -3, 0, 6, 0, -6, 0, -15, 0, 15, 0, 12, 0, -12, 0, -3, 0, 3, 0, -2, 0, 2, 0, 5, 0, -5, 0, -4, 0, 4, 0, 1, 0, -1, 0},
				 {-4, 10, -8, 2, 10, -25, 20, -5, -8, 20, -16, 4, 2, -5, 4, -1, 12, -30, 24, -6, -30, 75, -60, 15, 24, -60, 48, -12, -6, 15, -12, 3, -12, 30, -24, 6, 30, -75, 60, -15, -24, 60, -48, 12, 6, -15, 12, -3, 4, -10, 8, -2, -10, 25, -20, 5, 8, -20, 16, -4, -2, 5, -4, 1},
				 {2, -6, 6, -2, -5, 15, -15, 5, 4, -12, 12, -4, -1, 3, -3, 1, -6, 18, -18, 6, 15, -45, 45, -15, -12, 36, -36, 12, 3, -9, 9, -3, 6, -18, 18, -6, -15, 45, -45, 15, 12, -36, 36, -12, -3, 9, -9, 3, -2, 6, -6, 2, 5, -15, 15, -5, -4, 12, -12, 4, 1, -3, 3, -1},
				 {0, 2, 0, 0, 0, -6, 0, 0, 0, 6, 0, 0, 0, -2, 0, 0, 0, -6, 0, 0, 0, 18, 0, 0, 0, -18, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, -18, 0, 0, 0, 18, 0, 0, 0, -6, 0, 0, 0, -2, 0, 0, 0, 6, 0, 0, 0, -6, 0, 0, 0, 2, 0, 0},
				 {-1, 0, 1, 0, 3, 0, -3, 0, -3, 0, 3, 0, 1, 0, -1, 0, 3, 0, -3, 0, -9, 0, 9, 0, 9, 0, -9, 0, -3, 0, 3, 0, -3, 0, 3, 0, 9, 0, -9, 0, -9, 0, 9, 0, 3, 0, -3, 0, 1, 0, -1, 0, -3, 0, 3, 0, 3, 0, -3, 0, -1, 0, 1, 0},
				 {2, -5, 4, -1, -6, 15, -12, 3, 6, -15, 12, -3, -2, 5, -4, 1, -6, 15, -12, 3, 18, -45, 36, -9, -18, 45, -36, 9, 6, -15, 12, -3, 6, -15, 12, -3, -18, 45, -36, 9, 18, -45, 36, -9, -6, 15, -12, 3, -2, 5, -4, 1, 6, -15, 12, -3, -6, 15, -12, 3, 2, -5, 4, -1},
				 {-1, 3, -3, 1, 3, -9, 9, -3, -3, 9, -9, 3, 1, -3, 3, -1, 3, -9, 9, -3, -9, 27, -27, 9, 9, -27, 27, -9, -3, 9, -9, 3, -3, 9, -9, 3, 9, -27, 27, -9, -9, 27, -27, 9, 3, -9, 9, -3, 1, -3, 3, -1, -3, 9, -9, 3, 3, -9, 9, -3, -1, 3, -3, 1}};

/* value needed for derivatives, and a lookup table is the easiest/fastest method */
double factorial[4] = {1.0, 1.0, 2.0, 6.0};

int clip(int x, int a)
{
	/* use of a set of ternary operators to bound a value x between 0 and a */
	return x > a - 1 ? a - 1 : (x < 0 ? 0 : x);
}

int ijk2n(int i, int j, int k)
{
	return (i + 4 * j + 16 * k);
}

double tricubic_eval(double a[64], double x, double y, double z)
{
	int i, j, k;
	double ret = (double)(0.0);

	/* TRICUBIC EVAL
	This is the short version of tricubic_eval. It is used to compute
	the value of the function at a given point (x,y,z). To compute
	partial derivatives of f, use the full version with the extra args.*/

	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			for (k = 0; k < 4; k++)
			{
				ret += a[ijk2n(i, j, k)] * pow(x, i) * pow(y, j) * pow(z, k);
			}
		}
	}
	return (ret);
}

double tricubic_eval_derivatives(double a[64], double x, double y, double z, double multi, int dx, int dy, int dz)
{
	int i, j, k;
	double ret = (double)(0.0);
	double factx, facty, factz;

	/* TRICUBIC EVAL FULL
	It is used to compute the value of the function or derivative at a given point (x,y,z).
	The extra arguments, dx, dy, dz give the order of the derivative in that direction.
	multi is the scaling multiplier needed so that the proper scaling can be applied.*/

	for (i = dx; i < 4; i++)
	{
		factx = factorial[i] / factorial[i - dx];
		for (j = dy; j < 4; j++)
		{
			facty = factx * factorial[j] / factorial[j - dy];
			for (k = dz; k < 4; k++)
			{
				factz = facty * factorial[k] / factorial[k - dz];
				ret += factz * multi * a[ijk2n(i, j, k)] * pow(x, i - dx) * pow(y, j - dy) * pow(z, k - dz);
				/* needs factorial inclusion */
			}
		}
	}
	return (ret);
}

int _compare_fun(const void *a, const void *b)
{
	return (**(int **)a - **(int **)b);
}

void int_argsort(int outvec[], int invec[], int len)
{
	int i;
	int **temp = malloc(len * sizeof(int *));

	/* temp is constructed to reference invec, temp will
	be the modified order. outvec is the difference
	from the start to provide indices for other matricies */

	for (i = 0; i < len; i++)
	{
		temp[i] = &invec[i];
	}

	qsort(temp, len, sizeof(int *), _compare_fun);
	for (i = 0; i < len; i++)
	{
		outvec[i] = (int)(temp[i] - &invec[0]);
	}
	free(temp);
}

void voxel(double fin[], double f[], int tempx0, int tempx1, int tempx2, int ix0, int ix1, int ix2)
{
	int findx, l, k, j, templ, tempk, tempj;
	findx = 0;

	for (j = tempx2 - 1; j < tempx2 + 3; j++)
	{
		tempj = clip(j, ix2);
		for (k = tempx1 - 1; k < tempx1 + 3; k++)
		{
			tempk = clip(k, ix1);
			for (l = tempx0 - 1; l < tempx0 + 3; l++)
			{
				templ = clip(l, ix0);
				fin[findx] = *(f + templ + ix0 * (tempk + ix1 * tempj));
				findx++;
			}
		}
	}
}

void tricubic_get_coeff_stacked(double a[64], double x[64])
{
	int i, j;

	for (i = 0; i < 64; i++)
	{
		a[i] = (double)(0.0);

		for (j = 0; j < 64; j++)
		{
			a[i] += A[i][j] * x[j];
		}

		a[i] = a[i] / 8;
		/* A is the combination of A_v2 and the proper derivative operator as ints (requires a division by 8)  */
		//printf(" %f %i \n",a[i],i);
	}
}

void reg_ev_energy(double val[],
				   double x0[], double x1[], double x2[],
				   double f[], double fx0[], double fx1[], double fx2[],
				   int ix0, int ix1, int ix2, int ix)
{

	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, a[64], fin[64];
	double *dx0 = malloc(ix * sizeof(double));
	double *dx1 = malloc(ix * sizeof(double));
	double *dx2 = malloc(ix * sizeof(double));
	double *tempx0 = malloc(ix * sizeof(double));
	double *tempx1 = malloc(ix * sizeof(double));
	double *tempx2 = malloc(ix * sizeof(double));
	int *pos = malloc(ix * sizeof(int));
	int *indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	/*generate indices*/
	for (i = 0; i < ix; i++)
	{
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double)clip((int)tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double)clip((int)tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double)clip((int)tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int)tempx0[i] + ix0 * ((int)tempx1[i] + ix1 * ((int)tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for (i = 0; i < ix; i++)
	{

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if (iter != pos[loc])
		{
			iter = pos[loc];
			voxel(fin, f, (int)tempx0[loc], (int)tempx1[loc], (int)tempx2[loc], ix0, ix1, ix2);
			tricubic_get_coeff_stacked(a, fin);
		}
		val[loc] = tricubic_eval(a, dx0[loc], dx1[loc], dx2[loc]);
	}

	free(dx0);
	free(dx1);
	free(dx2);
	free(pos);
	free(indx);
	free(tempx0);
	free(tempx1);
	free(tempx2);
}

void reg_ev_forces(double val_dx0[], double val_dx1[], double val_dx2[],
				   double x0[], double x1[], double x2[],
				   double f[], double fx0[], double fx1[], double fx2[],
				   int ix0, int ix1, int ix2, int ix)
{

	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, N_dx0, N_dx1, N_dx2, a[64], fin[64];
	double *dx0 = malloc(ix * sizeof(double));
	double *dx1 = malloc(ix * sizeof(double));
	double *dx2 = malloc(ix * sizeof(double));
	double *tempx0 = malloc(ix * sizeof(double));
	double *tempx1 = malloc(ix * sizeof(double));
	double *tempx2 = malloc(ix * sizeof(double));
	int *pos = malloc(ix * sizeof(int));
	int *indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	N_dx0 = pow(dx0gap, -1);
	N_dx1 = pow(dx1gap, -1);
	N_dx2 = pow(dx2gap, -1);

	/*generate indices*/
	for (i = 0; i < ix; i++)
	{
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double)clip((int)tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double)clip((int)tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double)clip((int)tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int)tempx0[i] + ix0 * ((int)tempx1[i] + ix1 * ((int)tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for (i = 0; i < ix; i++)
	{

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if (iter != pos[loc])
		{
			iter = pos[loc];
			voxel(fin, f, (int)tempx0[loc], (int)tempx1[loc], (int)tempx2[loc], ix0, ix1, ix2);
			tricubic_get_coeff_stacked(a, fin);
		}
		val_dx0[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx0, 1, 0, 0);
		val_dx1[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx1, 0, 1, 0);
		val_dx2[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx2, 0, 0, 1);
	}

	free(dx0);
	free(dx1);
	free(dx2);
	free(pos);
	free(indx);
	free(tempx0);
	free(tempx1);
	free(tempx2);
}

void reg_ev_all(double val[], double val_dx0[], double val_dx1[], double val_dx2[],
				double x0[], double x1[], double x2[],
				double f[], double fx0[], double fx1[], double fx2[],
				int ix0, int ix1, int ix2, int ix)
{

	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, N_dx0, N_dx1, N_dx2, a[64], fin[64];
	double *dx0 = malloc(ix * sizeof(double));
	double *dx1 = malloc(ix * sizeof(double));
	double *dx2 = malloc(ix * sizeof(double));
	double *tempx0 = malloc(ix * sizeof(double));
	double *tempx1 = malloc(ix * sizeof(double));
	double *tempx2 = malloc(ix * sizeof(double));
	int *pos = malloc(ix * sizeof(int));
	int *indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	N_dx0 = pow(dx0gap, -1);
	N_dx1 = pow(dx1gap, -1);
	N_dx2 = pow(dx2gap, -1);

	/*generate indices*/
	for (i = 0; i < ix; i++)
	{
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double)clip((int)tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double)clip((int)tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double)clip((int)tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int)tempx0[i] + ix0 * ((int)tempx1[i] + ix1 * ((int)tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for (i = 0; i < ix; i++)
	{

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if (iter != pos[loc])
		{
			iter = pos[loc];
			voxel(fin, f, (int)tempx0[loc], (int)tempx1[loc], (int)tempx2[loc], ix0, ix1, ix2);
			tricubic_get_coeff_stacked(a, fin);
		}
		val[loc] = tricubic_eval(a, dx0[loc], dx1[loc], dx2[loc]);
		val_dx0[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx0, 1, 0, 0);
		val_dx1[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx1, 0, 1, 0);
		val_dx2[loc] = tricubic_eval_derivatives(a, dx0[loc], dx1[loc], dx2[loc], N_dx2, 0, 0, 1);
	}

	free(dx0);
	free(dx1);
	free(dx2);
	free(pos);
	free(indx);
	free(tempx0);
	free(tempx1);
	free(tempx2);
}
