#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "_tricube.h"


/*****************************************************************

    Based on eqtools package.
    
    https://github.com/PSFCPlasmaTools/eqtools

******************************************************************/

int clip(int x, int a){
	/* use of a set of ternary operators to bound a value x between 0 and a */
	return x > a - 1 ? a - 1 : (x < 0 ? 0 : x);
}

int ijk2n(int i, int j, int k){
    //return i + 4 * j + 16 * k;
	return i * 4 * 4 + j * 4 + k;
}


double tricubic_eval(double a[64], double x, double y, double z){
	int i, j, k;
	double ret = 0.0;

	/* TRICUBIC EVAL
	This is the short version of tricubic_eval. It is used to compute
	the value of the function at a given point (x,y,z). To compute
	partial derivatives of f, use the full version with the extra args.*/

	for(i = 0; i < 4; i++){
		for(j = 0; j < 4; j++){
			for(k = 0; k < 4; k++){
				ret += a[ijk2n(i, j, k)] * pow(x, i) * pow(y, j) * pow(z, k);
			}
		}
	}
	return ret;
}


double tricubic_eval_derivatives(double a[64], double x, double y, double z, double multi, int dx, int dy, int dz){
	int i, j, k;
	double ret = 0.0;
	double factx, facty, factz;

	/* TRICUBIC EVAL FULL
	It is used to compute the value of the function or derivative at a given point (x,y,z).
	The extra arguments, dx, dy, dz give the order of the derivative in that direction.
	multi is the scaling multiplier needed so that the proper scaling can be applied.*/

	for(i = dx; i < 4; i++){

		factx = factorial[i] / factorial[i-dx];
		for(j = dy; j < 4; j++){

			facty = factx * factorial[j] / factorial[j-dy];
			for(k = dz; k < 4; k++){

				factz = facty * factorial[k] / factorial[k-dz];
				ret += factz * multi * a[ijk2n(i, j, k)] * pow(x, i-dx) * pow(y, j-dy) * pow(z, k-dz);
				/* needs factorial inclusion */
			}
		}
	}
	return ret;
}


int _compare_fun(const void* a, const void* b){
	return (**(int**)a - **(int**)b);
}


void int_argsort(int outvec[], int invec[], int len){
	int i;
	int** temp = malloc(len * sizeof(int*));

	/* temp is constructed to reference invec, temp will
	be the modified order. outvec is the difference
	from the start to provide indices for other matricies */

	for(i = 0; i < len; i++){
		temp[i] = &invec[i];
	}

	qsort(temp, len, sizeof(int*), _compare_fun);
	for(i = 0; i < len; i++){
		outvec[i] = (int) (temp[i] - &invec[0]);
	}
	free(temp);
}


void voxel(double fin[], double f[], int tempx0, int tempx1, int tempx2, int ix0, int ix1, int ix2){
	int findx, tempi, tempj, tempk;
	int i, j, k;
	findx = 0;

	for(i = tempx0 - 1; i < tempx0 + 3; i++){
		tempi = clip(i, ix0);

		for(j = tempx1 - 1; j < tempx1 + 3; j++){
			tempj = clip(j, ix1);

			for(k = tempx2 - 1; k < tempx2 + 3; k++){
				tempk = clip(k, ix2);
				fin[findx] = *(f + tempi * ix1 * ix2 + tempj * ix2 + tempk);
				findx++;
			}
		}
	}
}


void tricubic_get_coeff_stacked(double a[64], double x[64]){
	int i, j;

	for(i = 0; i < 64; i++){
		a[i] = 0.0;

		for(j = 0; j < 64; j++){
			a[i] += A[i][j] * x[j];
		}

		a[i] = a[i]/8;
		/* A is the combination of A_v2 and the proper derivative operator as ints (requires a division by 8)  */
        //printf(" %f %i \n",a[i],i);
	}
}


void reg_ev_energy(double* val,
    double* x0, double* x1, double* x2,
    double* f, double* fx0, double* fx1, double* fx2,
    int ix0, int ix1, int ix2, int ix){


	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, a[64], fin[64];
	double* dx0 = malloc(ix * sizeof(double));
	double* dx1 = malloc(ix * sizeof(double));
	double* dx2 = malloc(ix * sizeof(double));
	double* tempx0 = malloc(ix * sizeof(double));
	double* tempx1 = malloc(ix * sizeof(double));
	double* tempx2 = malloc(ix * sizeof(double));
	int* pos = malloc(ix * sizeof(int));
	int* indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	/*generate indices*/
	for(i = 0; i < ix; i++){
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double) clip((int) tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double) clip((int) tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double) clip((int) tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int) tempx0[i] + ix0 * ((int) tempx1[i] + ix1 * ((int) tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for(i = 0; i < ix; i++){

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if(iter != pos[loc]){
			iter = pos[loc];
			voxel(fin, f, (int) tempx0[loc], (int) tempx1[loc], (int) tempx2[loc], ix0, ix1, ix2);
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



void reg_ev_forces(double* val_dx0, double* val_dx1, double* val_dx2,
	double* x0, double* x1, double* x2,
	double* f, double* fx0, double* fx1, double* fx2,
	int ix0, int ix1, int ix2, int ix){

	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, N_dx0, N_dx1, N_dx2, a[64], fin[64];
	double* dx0 = malloc(ix * sizeof(double));
	double* dx1 = malloc(ix * sizeof(double));
	double* dx2 = malloc(ix * sizeof(double));
	double* tempx0 = malloc(ix * sizeof(double));
	double* tempx1 = malloc(ix * sizeof(double));
	double* tempx2 = malloc(ix * sizeof(double));
	int* pos = malloc(ix * sizeof(int));
	int* indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	N_dx0 = pow(dx0gap, -1);
	N_dx1 = pow(dx1gap, -1);
	N_dx2 = pow(dx2gap, -1);


	/*generate indices*/
	for(i = 0; i < ix; i++){
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double) clip((int) tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double) clip((int) tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double) clip((int) tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int) tempx0[i] + ix0 * ((int) tempx1[i] + ix1 * ((int) tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for(i = 0; i < ix; i++){

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if(iter != pos[loc]){
			iter = pos[loc];
			voxel(fin, f, (int) tempx0[loc], (int) tempx1[loc], (int) tempx2[loc], ix0, ix1, ix2);
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


void reg_ev_all(double* val, double* val_dx0, double* val_dx1, double* val_dx2,
	double* x0, double* x1, double* x2,
	double* f, double* fx0, double* fx1, double* fx2,
	int ix0, int ix1, int ix2, int ix){

	int i, iter = -1, loc;
	double dx0gap, dx1gap, dx2gap, temp, N_dx0, N_dx1, N_dx2, a[64], fin[64];
	double* dx0 = malloc(ix * sizeof(double));
	double* dx1 = malloc(ix * sizeof(double));
	double* dx2 = malloc(ix * sizeof(double));
	double* tempx0 = malloc(ix * sizeof(double));
	double* tempx1 = malloc(ix * sizeof(double));
	double* tempx2 = malloc(ix * sizeof(double));
	int* pos = malloc(ix * sizeof(int));
	int* indx = malloc(ix * sizeof(int));

	dx0gap = fx0[1] - fx0[0];
	dx1gap = fx1[1] - fx1[0];
	dx2gap = fx2[1] - fx2[0];

	N_dx0 = pow(dx0gap, -1);
	N_dx1 = pow(dx1gap, -1);
	N_dx2 = pow(dx2gap, -1);


	/*generate indices*/
	for(i = 0; i < ix; i++){
		temp = (x0[i] - fx0[0]) / dx0gap;
		dx0[i] = modf(temp, &tempx0[i]);
		tempx0[i] = (double) clip((int) tempx0[i], ix0);
		dx0[i] = temp - tempx0[i];

		temp = (x1[i] - fx1[0]) / dx1gap;
		dx1[i] = modf(temp, &tempx1[i]);
		tempx1[i] = (double) clip((int) tempx1[i], ix1);
		dx1[i] = temp - tempx1[i];

		temp = (x2[i] - fx2[0]) / dx2gap;
		dx2[i] = modf(temp, &tempx2[i]);
		tempx2[i] = (double) clip((int) tempx2[i], ix2);
		dx2[i] = temp - tempx2[i];

		pos[i] = (int) tempx0[i] + ix0 * ((int) tempx1[i] + ix1 * ((int) tempx2[i]));
	}

	// find the right order for the evaluation to try and save time
	int_argsort(indx, pos, ix);

	for(i = 0; i < ix; i++){

		/* generate matrix for input into interp, this
		is the first attempt at trying to speed up the
		equation by forcing it more onto the C side*/

		loc = indx[i];
		if(iter != pos[loc]){
			iter = pos[loc];
			voxel(fin, f, (int) tempx0[loc], (int) tempx1[loc], (int) tempx2[loc], ix0, ix1, ix2);
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

