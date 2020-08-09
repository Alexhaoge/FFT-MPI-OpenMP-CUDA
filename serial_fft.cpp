//serial fft second version
//use malloc/calloc instead of global variables
//thus avoiding the problem of compile memory exceed when n=10^7
//otherwise option "-mcmodel=large" is needed when compile
//so memory allocation might take some time
//original author:  fnoi12bzzhan
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
using namespace std;
typedef complex<double> cpx;
int const N=10000100,M=34000000;
double const PI=acos(-1);
int n,t; char *s;
cpx *a,*b,*c;
int *ans,*pos;
void FFT(cpx x[],int f)
{
    for(int i=0;i<t;i++) if(i<pos[i]) swap(x[i],x[pos[i]]);
    for(int i=1;i<t;i<<=1)
    {
        cpx Wn=cpx(cos(PI/i),f*sin(PI/i));
        for(int j=0;j<t;j+=i<<1)
        {
            cpx w=1;
            for(int k=0;k<i;k++,w*=Wn)
            {
                cpx p=x[j+k],q=w*x[j+k+i];
                x[j+k]=p+q,x[j+k+i]=p-q;
            }
        }
    }
    if(f==-1) for(int i=0;i<t;i++) x[i]/=t;
}
void init(){
    s = (char*)calloc(n+10, sizeof(char));
    //must (char*) to announce the type
    a = (cpx*)calloc(t,sizeof(cpx));
    b = (cpx*)calloc(t,sizeof(cpx));
    c = (cpx*)calloc(t,sizeof(cpx));
    ans = (int*)calloc(t,sizeof(int));
    pos = (int*)calloc(t,sizeof(int));
}
void del(){
    free(s);free(a);free(b);
    free(c);free(ans);free(pos);
}
struct timespec time1 = {0, 0};
struct timespec time2 = {0, 0};
int main()
{
	clock_t beg=clock();
    freopen("fft.in","r",stdin);
    FILE *out=fopen("sfft.out","w");
    scanf("%d",&n);
    t=1; int n0=0; while(t<=n+n) t<<=1,n0++;
    init();
    scanf("%s",s); for(int i=0;i<n;i++) a[i]=s[n-i-1]-'0';
    scanf("%s",s); for(int i=0;i<n;i++) b[i]=s[n-i-1]-'0';
    for(int i=0;i<t;i++) pos[i]=(pos[i>>1]>>1)|((i&1)<<(n0-1));
    FFT(a,1),FFT(b,1);
    for(int i=0;i<t;i++) c[i]=a[i]*b[i];
    FFT(c,-1);
    for(int i=0;i<t;i++) ans[i]=(int)(c[i].real()+0.5);
    for(int i=0;i<t;i++) ans[i+1]+=ans[i]/10,ans[i]%=10;
    while(!ans[t]) t--;
    for(int i=t;i>=0;i--) fprintf(out,"%d",ans[i]);
    del();
    clock_t end=clock();
    printf("%lf\t", (double)(end-beg)/CLOCKS_PER_SEC*1000.0);
    return 0;
}
