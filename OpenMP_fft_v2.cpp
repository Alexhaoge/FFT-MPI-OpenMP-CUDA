//omp second version with doacross
//using calloc to avoid MLE during compilation, may cost time
#include <complex>
#include <cstdio>
#include <ctime>
#include <cstdlib> 
#include <string>
#include <sstream>
#include "omp.h"
using namespace std;
typedef complex<double> cpx;
int const N=10000000, M=34000000;
double const PI=acos(-1);
int n,t,threads=4; char *s;
cpx *a,*b,*c,*buf,*omg,*inv;
int *pos,*ans;
void mem(){
    s=(char*)calloc(n+10,sizeof(char));
    a=(cpx*)calloc(t,sizeof(cpx));
    b=(cpx*)calloc(t,sizeof(cpx));
    c=(cpx*)calloc(t,sizeof(cpx));
    buf=(cpx*)calloc(t,sizeof(cpx));
    omg=(cpx*)calloc(t,sizeof(cpx));
    inv=(cpx*)calloc(t,sizeof(cpx));
    ans=(int*)calloc(t,sizeof(int));
    pos=(int*)calloc(t,sizeof(int));
}
void del(){
    free(a);free(b);free(c);
    free(buf);free(omg);free(inv);
    free(ans);free(pos);
}
void init(){
    int i;
    #pragma omp parallel for shared(omg,inv,t)
    for(i = 0; i < t; i++){
        omg[i] = cpx(cos(2*PI*i/t),sin(2*PI*i/t));
        inv[i] = conj(omg[i]);
    }
}
void FFT(cpx x[],bool f,cpx om[])//f==1 IDFT, f==0 DFT
{
    int i,tmp,j,k;cpx t1;
    #pragma omp parallel for shared(x,pos)
    for(i=0;i<t;i++)
    	if(i<pos[i])
    		swap(x[i],x[pos[i]]);
    for(i=1;i<t;i<<=1)
    {
        tmp = i<<1;
        #pragma omp parallel shared(x,buf,om,i,t,tmp) private(j,k,t1)
        {
        	#pragma omp for 
        	for(j=0;j<t;j+=tmp)//doacross alternative
        	{
            	for(k=0;k<i;k++)
            	{
            		t1=om[t/tmp*k]*x[j+k+i];
            	    buf[j+k]=x[j+k]+t1;
            	    buf[j+k+i]=x[j+k]-t1;
            	}
        	}
        	#pragma omp for
        	for(j=0;j<t;j++)
            	x[j] = buf[j];
		}
    }
    if(f)
        #pragma omp parallel for shared(x)
        for(i=0;i<t;i++) x[i]/=t;
}
int main(int argc, char **argv){
    double beg=omp_get_wtime();
    stringstream ss;
    ss<<argv[1];
    ss>>threads;
    omp_set_num_threads(threads);
    freopen("fft.in","r",stdin);
    FILE *out=fopen("fft2.out","w");
    scanf("%d",&n);
    t=1; int n0=0; while(t<n+n) t<<=1,n0++;
    mem();
    scanf("%s",s); for(int i=0;i<n;i++) a[i]=s[n-i-1]-'0';
    scanf("%s",s); for(int i=0;i<n;i++) b[i]=s[n-i-1]-'0';
    int i;init();free(s);
    for(i=0;i<t;i++) pos[i]=(pos[i>>1]>>1)|((i&1)<<(n0-1));
    FFT(a,0,omg);
    FFT(b,0,omg);
    //parallel for nesting with section is not allowed!!!
    #pragma omp parallel for shared(c,a,b)
    for(i=0;i<t;i++) c[i]=a[i]*b[i];
    FFT(c,1,inv);
    #pragma omp parallel for shared(ans,c)
    for(i=0;i<t;i++) ans[i]=(int)(c[i].real()+0.5);
    for(i=0;i<t;i++) ans[i+1]+=ans[i]/10,ans[i]%=10;
    while(!ans[t]) t--;
    for(i=t;i>=0;i--) fprintf(out,"%d",ans[i]);
    del();
    double end=omp_get_wtime();
    printf("%lf", (end-beg)*1000.0);
    return 0;
}
