//omp initial version without doacross
//when N>=10^7 use compile option -mcmodel=large
#include <complex>
#include <cstdio>
#include <ctime>
#include <string>
#include <sstream>
#include "omp.h"
using namespace std;
typedef complex<double> cpx;
int const N=1100000, M=2500000;
double const PI=acos(-1);
int n,t,threads=4; char s[N];
cpx a[M],b[M],c[M],buf[M],omg[M],inv[M];
int pos[M];
void init(){
    int i;
    #pragma omp parallel for shared(omg)
    for(i = 0; i < t; i++){
        omg[i] = cpx(cos(2*PI*i/t),sin(2*PI*i/t));
    }
    #pragma omp parallel for shared(inv,omg)
    for(i = 0; i < t; i++)
        inv[i] = conj(omg[i]);
}
void FFT(cpx x[],bool f,cpx om[])//f==1 IDFT
{
    int i,tmp,j,k;cpx t1;
    #pragma omp parallel for shared(x,pos)
    for(i=0;i<t;i++)
    	if(i<pos[i])
    		swap(x[i],x[pos[i]]);
    for(i=1;i<t;i<<=1)
    {
        tmp = i<<1;
        #pragma omp parallel for shared(x,buf,om,tmp,i,t) private(j,k,t1)
		for(j=0;j<t;j+=tmp)
        {
            for(k=0;k<i;k++)
            {
            	t1=om[t/tmp*k]*x[j+k+i];
                buf[j+k]=x[j+k]+t1;
                buf[j+k+i]=x[j+k]-t1;
                //printf("%d %d %d\n",i,j,k);
            }
        }
        #pragma omp parallel for shared(x,buf,t)
        for(j=0;j<t;j++)
            x[j] = buf[j];
    }
    if(f)
        #pragma omp parallel for shared(x)
        for(i=0;i<t;i++) x[i]/=t;
    //for(i=0;i<t;i++)
    //    printf("%lf %lf\n",x[i].real(),x[i].imag());
    //printf("-----\n");
}
int ans[M];
int main(int argc, char **argv){
    double beg=omp_get_wtime();
    stringstream ss;
    ss<<argv[1];
    ss>>threads;
    omp_set_num_threads(threads);
    freopen("fft.in","r",stdin);
    FILE *out=fopen("fft1.out","w");
    scanf("%d",&n);
    scanf("%s",s); for(int i=0;i<n;i++) a[i]=s[n-i-1]-'0';
    scanf("%s",s); for(int i=0;i<n;i++) b[i]=s[n-i-1]-'0';
    t=1; int n0=0; while(t<n+n) t<<=1,n0++;
    //printf("%d\n",t);
    int i;init();
    for(i=0;i<t;i++) pos[i]=(pos[i>>1]>>1)|((i&1)<<(n0-1));
    //#pragma omp parallel sections
    {
        //#pragma omp section
            FFT(a,0,omg);
        //#pragma omp section
            FFT(b,0,omg);
    }//parallel for nesting with section is not allowed!!!
    #pragma omp parallel for shared(c,a,b)
    for(i=0;i<t;i++) c[i]=a[i]*b[i];
    FFT(c,1,inv);
    #pragma omp parallel for shared(ans,c)
    for(i=0;i<t;i++) ans[i]=(int)(c[i].real()+0.5);
    for(i=0;i<t;i++) ans[i+1]+=ans[i]/10,ans[i]%=10;
    while(!ans[t]) t--;
    for(i=t;i>=0;i--) fprintf(out,"%d",ans[i]);
    double end=omp_get_wtime();
    printf("%lf", (end-beg)*1000.0);
    return 0;
}
