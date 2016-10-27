
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>

using namespace std;

int main(int argc, char* argv[])
{
    if(argc<4)
    {
        printf("Usage: %s input_filename positive_output negative_output group_val\n", argv[0]);
        printf("Here the file has to be one of the JAIST Nucleosome dataset file.\n");
        return -1;
    }
    
    char str[600];
    char value[10];
    char ch;
    
    int group_val = atoi(argv[4]);
    
    FILE *fp1;
    FILE *fp2;
    FILE *fp3;
    
    fp1 = fopen(argv[1],"r");\
    if(fp1==NULL)
    {
        printf("Error Opening File: %s\n", argv[1]);
        return -5;
    }
    fp2 = fopen(argv[2],"w");
    if(fp2==NULL)
    {
        fclose(fp1);
        printf("Error Opening File: %s\n", argv[2]);
        return -6;
    }
    fp3 = fopen(argv[3],"w");
    if(fp3==NULL)
    {
        fclose(fp1);
        fclose(fp2);
        printf("Error Opening File: %s\n", argv[3]);
        return -7;
    }
    while((ch = fgetc(fp1)) != EOF)
    {
        if(fgets(str,600,fp1)==NULL)
        {
            printf("Error Reading File! - 1\n");
            fclose(fp1);
            fclose(fp2);
            fclose(fp3);
            return -2;
        }
        
        if(fgets(str,600,fp1)==NULL)
        {
            printf("Error Reading File! - 2\n");
            fclose(fp1);
            fclose(fp2);
            fclose(fp3);
            return -3;
        }
        
        if(fgets(value,10,fp1)==NULL)
        {
            printf("Error Reading File! - 3\n");
            fclose(fp1);
            fclose(fp2);
            fclose(fp3);
            return -4;
        }
        
        if(atoi(value)==1)
        {
            for(int i=0; str[i+group_val]!='\n'; i++)
            {
                for(int j=0; j<group_val; j++)
                    fputc(str[i+j],fp2);
                
                fputc(' ',fp2);
            }
            fputc('\n',fp2);
        }
        else
        {
            for(int i=0; str[i+group_val]!='\n'; i++)
            {
                for(int j=0; j<group_val; j++)
                    fputc(str[i+j],fp3);
                    
                fputc(' ',fp3);
            }
            fputc('\n',fp3);
        }
    }
    
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    return 0;
}
