#include <omp.h>
#include <stdio.h>

int main()
{
#pragma omp parallel
{
    int ID=0;
    printf("hello(%d)\n", ID);
}
}
