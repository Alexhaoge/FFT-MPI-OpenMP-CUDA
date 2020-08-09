//Attempt to solve FFT when n>10^7 using MPI with distributed memory 
//I try to save memory instead of time because when n large enough
//the memory may not be enough to store the whole array in a single processor
//So in this program, each processor only store a part of the array
//MUST meet these requirements:
//1. assign 2^n processors
//2. processors <= n/processor
//3. n/processor <= 2^30
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <complex>
#include <algorithm>
#include <cmath>
#include "mpi.h"
using namespace std;
typedef complex<double> cpx;
double const PI = acos(-1);
long long n, T;
int id, sz, len, *pos, *p_pos;
char* str;
cpx* a, *b, *c;
long long* ans;
FILE *in;
MPI_Status status;
inline cpx omega(long long k, long long nn, int f) {
	return cpx(cos(PI * k / nn), f * sin(PI * k / nn));
}
void init() {
	a = (cpx*)calloc(len, sizeof(cpx));
	b = (cpx*)calloc(len, sizeof(cpx));
	str = (char*)calloc(sz, sizeof(char));
	pos = (int*)calloc(len, sizeof(int));
	int n0 = (int)log2(len);
	for (int i = 0; i < len; i++)
		pos[i] = (pos[i >> 1] >> 1) | ((i & 1) << (n0 - 1));
	p_pos = (int*)calloc(sz, sizeof(int));
	n0 = (int)log2(sz);
	for (int i = 0; i < sz; i++)
		p_pos[i] = (p_pos[i >> 1] >> 1) | ((i & 1) << (n0 - 1));
}
inline void readin(cpx* aa) {
	char* tmp = (char*)calloc(sz, sizeof(char)), hc;
	if (id == 0)	fscanf(in, "\n");
	int n1 = n / sz, i;
	for (i = len - 1; i > n1; i--)	aa[i] = 0.0;
	int residue = n % sz;
	if (id == 0)	fread(tmp, sizeof(char), residue, in);
	MPI_Bcast(tmp, residue, MPI_CHAR, 0, MPI_COMM_WORLD);
	if (p_pos[id] >= residue)	aa[n1] = 0.0;
	else  aa[n1] = tmp[residue - p_pos[id] - 1] - '0';
	for (i = n1 - 1; i >= 0; i--) {
		if (id == 0) fread(tmp, sizeof(char), sz, in);
		MPI_Bcast(tmp, sz, MPI_CHAR, 0, MPI_COMM_WORLD);
		aa[i] = tmp[sz - p_pos[id] - 1] - '0';
	}
}
void local_fft(cpx* x, int f) {
	cpx Wn, w, p, q;
	int i, j, k;
	for (i = 0; i < len; i++) if (i < pos[i]) swap(x[i], x[pos[i]]);
	for (i = 1; i < len; i <<= 1) {
		Wn = cpx(cos(PI / i), f * sin(PI / i));
		for (j = 0; j < len; j += i << 1) {
			w = 1;
			for (k = 0; k < i; k++, w *= Wn) {
				p = x[j + k], q = w * x[j + k + i];
				x[j + k] = p + q, x[j + k + i] = p - q;
			}
		}
	}
}
inline void butterfly(cpx* x, cpx* tmp, int pa, int pb, long long i, long long j, int f) {
	int counterpart = id == pa ? pb : pa;
	MPI_Sendrecv(x, len, MPI_C_DOUBLE_COMPLEX, counterpart, id, tmp,
		len, MPI_C_DOUBLE_COMPLEX, counterpart, counterpart, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
	cpx w;
	for (int k = 0; k < len; k++) {
		w = omega(j + k, i, f);
		if (id == pa)	x[k] = x[k] + w * c[k];
		else if (id == pb)	x[k] = c[k] - w * x[k];
	}
}
void FFT(cpx* x, cpx* tmp, int f) {//f==1 FFT, f==-1 IFFT
	local_fft(x, f);
	// printf("%d: fft_local ", id);
	// for(int i=0;i<len;i++) printf("%.0lf %.0lf    ",x[i].real(),x[i].imag());
	// printf("\n");
	long long f1 = 1;
	for (long long i = len; i < T; i <<= 1) {
		int step = i / len, pa;
		if (id % (step << 1) >= step) pa = id - step;
		else pa = id;
		butterfly(x, tmp, pa, pa + step, i, f1*pa*len % (i << 1), f);
	}
	if (f == -1)
		for (int i = 0; i < len; i++)   x[i] /= T;
}
void mul(cpx* aa, cpx* bb) {
	for (int i = 0; i < len; i++) aa[i] = aa[i] * bb[i];
}
void Bitwise(cpx *aa, cpx *tmp) {
	int zoom = len / sz;
	for (int i = 0; i < len; i++)	tmp[zoom* p_pos[i%sz] + i / sz] = aa[i];
	MPI_Alltoall(tmp, zoom, MPI_C_DOUBLE_COMPLEX, aa, zoom, MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
}
int convert(cpx* aa, long long* an) {
	long long jw = 0;
	for (int i = 0; i < len; i++)
		an[i] = (long long)(aa[i].real() + 0.5);
	if (id) MPI_Recv(&jw, 1, MPI_LONG_LONG, id - 1, id - 1, MPI_COMM_WORLD, &status);
	an[0] += jw;
	int zgw = an[0] == 0 ? 0 : id, global_zgw = 0;
	for (int i = 0; i < len - 1; i++) {
		an[i + 1] += an[i] / 10, an[i] %= 10;
		if (!zgw && an[i + 1] != 0) zgw = id;
	}
	jw = an[len - 1] / 10, an[len - 1] %= 10;
	if (id != sz - 1)
		MPI_Send(&jw, 1, MPI_LONG_LONG, id + 1, id, MPI_COMM_WORLD);
	MPI_Allreduce(&zgw, &global_zgw, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	return global_zgw;
}
void print(long long* aa, int zgw) {
	int flag = id, i; MPI_Status status;
	if (id == zgw) {
		for (i = len - 1; i >= 0 && !ans[i]; i--);
		FILE* out = fopen("mpifft.out", "w");
		for (; i >= 0; i--) fprintf(out, "%lld", ans[i]);
		fclose(out);
		if (id) MPI_Send(&flag, 1, MPI_INT, id - 1, id, MPI_COMM_WORLD);
	}
	else if (id < zgw) {
		MPI_Recv(&flag, 1, MPI_INT, id + 1, id + 1, MPI_COMM_WORLD, &status);
		FILE* out = fopen("mpifft.out", "a");
		for (int i = len - 1; i >= 0; i--)
			fprintf(out, "%lld", ans[i]);
		fclose(out);
		if (id) MPI_Send(&flag, 1, MPI_INT, id - 1, id, MPI_COMM_WORLD);
	}
}

int main(int argc, char **argv) {
	//init
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);
	double beg = MPI_Wtime();
	if (id == 0) {
		in = fopen("fft.in", "r");
		fscanf(in, "%lld", &n);
	}
	MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
	T = 1; while (T <= n + n) T <<= 1;
	if (T % sz) {
		printf("%d: Processor number must be 2^n, found to be %d\n", id, sz);
		MPI_Finalize();
		return 1;
	}
	len = T / sz;
	init(); readin(a); readin(b);
	free(str);
	c = (cpx*)calloc(len, sizeof(cpx));
	FFT(a, c, 1);
	FFT(b, c, 1);
	mul(a, b);
	free(b);
	Bitwise(a, c);
	FFT(a, c, -1);
	free(c), free(pos), free(p_pos);
	ans = (long long*)calloc(len, sizeof(long long));
	int zgw = convert(a, ans); free(a);
	print(ans, zgw);
	free(ans);
	MPI_Barrier(MPI_COMM_WORLD);
	double end = MPI_Wtime();
	if (id == 0) printf("%lf\t", (end - beg) * 1000);
	MPI_Finalize();
	return 0;
}
