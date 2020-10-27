/*
 * main.cpp
 *
 *  Created on: 09.01.2015
 *      Author: alexey slovesnov
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <map>
#include "reversi/Reversi.h"

//#define ZERO_WINDOW

void solve6x6(bool maximize);
void solve5x5(bool maximize);
void solve4x4(bool maximize);
void test();
void dataToHtml();
void solve6x6File(bool maximize);

int main (int argc, char *argv[]) {
	if(argc==5){//multisolver
#ifdef USE_FILE_TO_SOLVE
		assert(0 && "USE_FILE_TO_SOLVE is defined");
#endif
		int tableSize=atoi(argv[2]);
		assert(tableSize==5 || tableSize==6);
		int thread=atoi(argv[3]);
		bool max=atoi(argv[4])!=0;
		ReversiBase*r=NULL;
		if(tableSize==6){
			if(max){
				r=new Reversi<6,true,true>();
			}
			else{
				r=new Reversi<6,true,false>();
			}
		}
		else if(tableSize==5){
			if(max){
				r=new Reversi<5,true,true>();
			}
			else{
				r=new Reversi<5,true,false>();
			}
		}
		else{
			printf("use %s file tablesize thread max\n",argv[0]);
		}
		assert(r!=0);
		r->prepare();
		r->solveFile(argv[1],thread);
		r->free();
		delete r;
	}
	else if(argc==2 || argc==3){//multisolver output info
		int n=argc==3 ? atoi(argv[2]): 0;
		Reversi<6,false>::solveFileInfo(argv[1],n);
	}
	else{
//		solve6x6(true);
//		solve6x6(false);
//		solve5x5(true);
//		solve5x5(false);
//		solve4x4(true);
//		solve4x4(false);
		solve6x6File(true);
//		dataToHtml();
		//test();
	}
	return 0;
}

void solve6x6(bool maximize){
	const bool storeCodes=false;
	int e;
	long t;
	FILE*f;
	clock_t beginProgram=clock();
#ifdef ZERO_WINDOW
		const int alpha=0;
#endif

#ifdef NODE_COUNT
	uint64_t total;
#endif
	ReversiBase*reversi;
	int empties,can;
	clock_t begin;
	const int VALUE_MAX[]={
			 16, -8,//number of empty cells 4
			-16,-30,//number of empty cells 5
			 20,  4,//number of empty cells 6
			 27, -4,//number of empty cells 7
			 20,  4,//number of empty cells 8
			  4,-24,//number of empty cells 9
			 28, 16,//number of empty cells 10
			 14, 16,//number of empty cells 11
			  4,-22,//number of empty cells 12
			-10,-20,//number of empty cells 13
			  8, 14,//number of empty cells 14
			 -2,  6,//number of empty cells 15
			-30,-16,//number of empty cells 16
			  0,-32,//number of empty cells 17
			 30, 14,//number of empty cells 18
			 -8,-10,//number of empty cells 19
			 20,-26,//number of empty cells 20
			 26,  2,//number of empty cells 21
			 30,-10,//number of empty cells 22
			 -6, 16,//number of empty cells 23
			-14, 13,//number of empty cells 24
			//number of empty cells 25
	};

	const int VALUE_MIN[]={
			 10,-16,//number of empty cells 4
			-10, -2,//number of empty cells 5
			  2,  4,//number of empty cells 6
			 16,  2,//number of empty cells 7
			 -2,  0,//number of empty cells 8
			 -6, -4,//number of empty cells 9
			  4, 10,//number of empty cells 10
			  2, -4,//number of empty cells 11
			 -6,  4,//number of empty cells 12
			  0, -4,//number of empty cells 13
			 12, 12,//number of empty cells 14
			  0,-12,//number of empty cells 15
			 -6,  2,//number of empty cells 16
			 -2,  2,//number of empty cells 17
			  4, -2,//number of empty cells 18
			 -8, -2,//number of empty cells 19
			 	0, -4,//number of empty cells 20
	};

	if(storeCodes){
		f=fopen("codes.txt","w+");
	}
	typedef Reversi<6,true,true> R6max;
	typedef Reversi<6,true,false> R6min;
	const int T2=R6max::tableSize2;
	const int*pvalue=maximize ? VALUE_MAX : VALUE_MIN;
	const int*pvalueStart=pvalue;
	const int VALUE_SIZE = maximize ? SIZE( VALUE_MAX) : SIZE(VALUE_MIN);//do not use SIZE(maximize ? VALUE_MAX : VALUE_MIN )
	if(maximize){
		R6max::create();
	}
	else{
		R6min::create();
	}
//	std::vector<int> v;
	for(empties=4;empties<(storeCodes ? 20 : 32);empties++){
		for(can=0;can<2;can++,pvalue++){
			if(maximize){
				reversi=&R6max::init(can==0 ? 0: 2,T2-4-empties,false);
			}
			else{
				reversi=&R6min::init(can==0 ? 0: 2,T2-4-empties,false);
			}
			//r.print();
			begin=clock();

			if(storeCodes){
				char c[128];
				fprintf(f,"%s%40s\n",_i64toa(reversi->hash(WHITE),c,10)," ");//Note Have to store Hash with symmetry 0
			}

#ifdef ZERO_WINDOW
			e=reversi->value(alpha,alpha+1,WHITE);
#else
			e=reversi->valueZeroWindow(-T2,T2,WHITE);
			//e=reversi->value(-T2,T2,WHITE);
#endif

//			v.push_back(e);
			if( pvalue-pvalueStart<VALUE_SIZE ){
//				printf("check ");
			if(
#ifdef ZERO_WINDOW
					(e<=alpha && *pvalue>=alpha+1 ) || (e>=alpha+1 && *pvalue<=alpha)
#else
			e!=*pvalue
#endif
			){
					printf("error %d %d %d\n",__LINE__,e,*pvalue);
					return;
				}
			}
			t=clock() - begin;
			printf("%02ld:%02ld.%03ld e=%3d empty=%2d c%d ",t/CLOCKS_PER_SEC/60,t/CLOCKS_PER_SEC%60, t%CLOCKS_PER_SEC
				, e	,empties, 1-can );


#ifdef NODE_COUNT
			total=0;
			if(maximize){
				for(e=0;e<R6max::maxDepth;e++){
					total+=R6max::nodeCount[e];
				}
			}
			else{
				for(e=0;e<R6min::maxDepth;e++){
					total+=R6min::nodeCount[e];
				}
			}
			printf("nodes %.2le %15s",double(total),reversi->uint64ToString(total).c_str() );
#endif
			t=clock()-beginProgram;
			printf(" %02ld:%02ld.%03ld\n",t/CLOCKS_PER_SEC/60,t/CLOCKS_PER_SEC%60, t%CLOCKS_PER_SEC);

//			for(std::vector<int>::const_iterator it=v.begin();it!=v.end();it++){
//				printf("%d, ",*it);
//			}
//			printf("\n");

			fflush(stdout);
		}

	}
	if(storeCodes){
		fclose(f);
	}

	t=(clock() - beginProgram)/CLOCKS_PER_SEC;
	printf("total time %02ld:%02ld\n",t/60,t%60);
	reversi->free();
}

void solve6x6File(bool maximize){
#ifndef USE_FILE_TO_SOLVE
	assert(0);
#endif
	int i;
	int type;
	ReversiBase*r;
	const int tableSize=6;
	const int tableSize2=tableSize*tableSize;
	assert(maximize==true);
	printf("solve%dx%d %s\n",tableSize,tableSize,maximize?"maximize":"minimize");
	fflush(stdout);
	for(type=0;type<6;type++){
		if(maximize){
			Reversi<tableSize,true,true>::create();
			r= &Reversi<tableSize,true,true>::init(type,0,false);
		}
		else{
			Reversi<tableSize,true,false>::create();
			r= &Reversi<tableSize,true,false>::init(type,0,false);
		}
		i=r->value(-tableSize2,tableSize2,BLACK);
		printf("type%d %d\n",type,i);
		fflush(stdout);
	}
	printf("\n");
	r->free();
}

void solve5x5(bool maximize){
	int i;
	int type;
	ReversiBase*r;
	const int tableSize=5;
	const int tableSize2=tableSize*tableSize;
	printf("solve%dx%d %s\n",tableSize,tableSize,maximize?"maximize":"minimize");
	fflush(stdout);
	for(type=0;type<6;type++){
		if(maximize){
			Reversi<tableSize,true,true>::create();
			r= &Reversi<tableSize,true,true>::init(type,0,false);
		}
		else{
			Reversi<tableSize,true,false>::create();
			r= &Reversi<tableSize,true,false>::init(type,0,false);
		}
		i=r->value(-tableSize2,tableSize2,BLACK);
		printf("type%d %d\n",type,i);
		fflush(stdout);
	}
	printf("\n");
	r->free();
}

void solve4x4(bool maximize){
	int i;
	int type;
	ReversiBase*r;
	const int tableSize=4;
	const int tableSize2=tableSize*tableSize;
	printf("solve%dx%d %s\n",tableSize,tableSize,maximize?"maximize":"minimize");
	fflush(stdout);
	for(type=0;type<6;type++){
		if(maximize){
			Reversi<tableSize,false,true>::create();
			r= &Reversi<tableSize,false,true>::init(type,0,false);
		}
		else{
			Reversi<tableSize,false,false>::create();
			r= &Reversi<tableSize,false,false>::init(type,0,false);
		}
//		i=r->value(0,1,BLACK);
//		i=r->valueZeroWindow(-tableSize2,tableSize2,BLACK);
		i=r->value(-tableSize2,tableSize2,BLACK);
		printf("type%d %d\n",type,i);
		fflush(stdout);
	}
	printf("\n");
}

void test(){
	const int SZ=6;
	typedef Reversi<SZ,false,true> Rev;
//	const int classes=2;
//	Rev::countAllNodesSymmetry(0,23, classes);

//	Rev::countAllNodes(0,0,false);
//	Rev::create();
//	Rev::storeNodes(0,5,16);

	Rev::create();
//	Rev::transformFileAfterSolution("min.txt","min.dat");
	Rev::transformFileAfterSolution("6max1.txt","6max24.dat");
//	Rev::storeNodes(0,5,24);
//	Rev::storeSNodes(24);

//	FILE*f=fopen("max.dat","rb");
//	assert(f!=NULL);
//	char c;
//	int minE=INT_MAX,maxE=INT_MIN;
//	int i=1;
//	while(fread(&c,1,1,f)==1){
//		if(c<minE){
//			minE=c;
//		}
//		if(c>maxE){
//			maxE=c;
//		}
//		i++;
//	}
//	printf("read %d [%d %d]\n",i,minE,maxE);
//	fclose(f);
}

std::string removeCommas(std::string s){
	std::string r;
	std::string::const_iterator its;
	for(its=s.begin();its!=s.end();its++){
		if(isdigit(*its)){//remove ','
			r+=*its;
		}
	}
	return r;
}

void dataToHtml(){
	const int BUFFER_SIZE=128;
	char c[BUFFER_SIZE];
	char*p,*p1;
	int i,j=-1,k,l;
	const int MAP_SIZE=12;
	typedef std::map<int,uint64_t> MT;
	MT map[MAP_SIZE];
	MT::const_reverse_iterator it,it1;
	std::vector<int> type;
	std::vector<int> symmetry;
	bool first;
	uint64_t total;
	std::vector<uint64_t> prev;
	std::string s;
	const char SYMMETRY[]="symmetry";
	const char TYPE[]="type";

	FILE*f=fopen("data.txt","r");
	assert(f!=NULL);

	while(fgets(c,BUFFER_SIZE,f)){
		first=true;
		if((p=strstr(c,TYPE))!=NULL && strstr(c,"countAllNodes")==c ){
			type.push_back(atoi(p+strlen(TYPE)));

			if(strstr(c,"countAllNodesSymmetry")==c){
				symmetry.push_back(1);
			}
			else{
				p=strstr(c,SYMMETRY);
				assert(p!=0);
				symmetry.push_back(atoi(p+strlen(SYMMETRY)));
			}
		}
		while( (isdigit(c[0]) || c[0]==' ') && isdigit(c[1]) && c[2]==' ' ){
			if(first){
				j++;
				first=false;
				assert(j<MAP_SIZE);
			}

			i=atoi(c);
			for(p=c+3;*p==' ';p++);
			p1=p;
			for( ; isdigit(*p) || *p==',' ; p++);
			*p=0;
			s=removeCommas(p1);
			map[j][i]+=_atoi64(s.c_str());//+= to sum layer classes
			fgets(c,BUFFER_SIZE,f);
		}
	}
	fclose(f);

	j++;
	assert(j%2==0);
	assert(int(type.size())==j);

	f=fopen("data.html","w+");
	assert(f!=NULL);
	fprintf(f,"<html><body>");
	for(i=0;i<j;){
		MT& m=map[i];
		for( k=i+1 ; k<j && (m.rbegin()->first==map[k].rbegin()->first) ; k++);

		prev.clear();
		for(l=i;l<k;l++){
			prev.push_back(1);
		}
		fprintf(f,"<table class=\"single\"><tr><td>&nbsp;");
		for(l=i;l<k;l++){
			fprintf(f,"<td align=\"center\"><div valign=\"top\">symmetry %d</div> <img src=\"img/reversi/start%d.png\">",symmetry[l],type[l]);
		}
		fprintf(f,"\n");
		for( it=m.rbegin() ; it!=m.rend() ; it++ ){
			fprintf(f,"<tr><td>%d",it->first);
			for(l=i;l<k;l++){
				total=map[l][it->first];
				fprintf(f,"<td align=\"right\">%s bf=%.2lf",ReversiBase::uint64ToString(total).c_str(),total/double(prev[l-i]) );
				prev[l-i]=total;
			}
			fprintf(f,"\n");
		}
		fprintf(f,"<tr><td>total");
		for(l=i;l<k;l++){
			total=0;
			for( it=map[l].rbegin() ; it!=map[l].rend() ; it++ ){
				total+=it->second;
			}
			fprintf(f,"<td align=\"right\">%s",ReversiBase::uint64ToString(total).c_str());
		}
		fprintf(f,"\n");

		printf("\n\n");

		fprintf(f,"</table>\n");
		i=k;
	}
	fprintf(f,"</body></html>");
	fclose(f);
}
