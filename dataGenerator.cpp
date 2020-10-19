#include <bits/stdc++.h>
using namespace std;
double ans[64];
char str[200][20]={"d15/mpi.log","d16/mpi.log",
"d17/mpi.log","d18/mpi.log","d19/mpi.log",	"d20/mpi.log",	"d21/mpi.log",	"d22/mpi.log",	"d23/mpi.log",	"d24/mpi.log",	"d25/mpi.log",	"d26/mpi.log",	"d27/mpi.log",	"d28/mpi.log",	"d29/mpi.log",	"d30/mpi.log",
};
int main(){
	freopen("a.out","w",stdout);
	double tmp;
	for(int idx=0;idx<16;idx++){
	FILE *file=fopen(str[idx],"r");
	for(int j=0;j<=(idx<=10?5:3);j++) fscanf(file,"%lf",&ans[j]);
	for(int i=0;i<7;i++) printf("%lf\n",ans[i]);
	printf("\n\n");
	}
} 
