/*
 * Reversi.h
 *
 *  Created on: 09.01.2015
 *      Author: alexey slovesnov
 *       Email: slovesnov@yandex.ru
 *
 * 	REVERSI END GAME SOLUTION
 *  supports different table sizes 3-8
 *  hash table supports maximum table size 6, for size 7-8 need to extend code
 *  supports 6 types of start position
 *  count all symmetry nodes for tables 4x4, 5x5 for all start positions, for table 6x6 count nodes partially
 *  count not symmetry nodes for table 6x6 partially
 *  found shortest games with only black chips, only white chips and both black and white chips for tables 4x4, 5x5, 6x6
 *
 *
 *  GOOD OPTIMIZATIONS (implemented)
 *  directions for do move
 *  static move ordering
 *  not do last move just count score
 *  using of list of empty cells (potentially moves)
 *  zero window search which is faster then normal alpha beta search
 *  hash table (using symmetry hash for hashKey is not good, so use simple hash code)
 *
 *
 *  BAD OPTIMIZATIONS (tried then removed)
 *  static game table[] array with undo move
 *  make table[] with size TABLE_SIZE2 not FULL_TABLE_SIZE2
 *  check next hash at first
 *
 *
 *	Notes
 *	Reversi is template class so all methods should be implemented in header file
 */

#ifndef REVERSI_H_
#define REVERSI_H_

#include <assert.h>
#include <stdint.h>
#include <string>
#include <string.h>
#include <algorithm>
#include <vector>
#include <map>
#include <limits.h>
#include <share.h>
#include "TurnItem.h"
#include "Hash.h"

//ifdef USE_FILE_TO_SOLVE then use precounted files to solve 5x5 and 6x6 tables
//USE_FILE_TO_SOLVE should be not defined for precount
#define USE_FILE_TO_SOLVE

//BEGIN basic defines & constants
#ifdef GTK_REVERSI
	//Note gtk application don't allow HASH_BITS=27 too big, HASH_BITS=25 works ok
	const int HASH_BITS=25;
#ifndef USE_FILE_TO_SOLVE
	#define USE_FILE_TO_SOLVE
#endif

#else
	const int HASH_BITS=27;
	#define println printf
#endif

const int TURNS_LEFT=8;//hash only at least TURNS_LEFT empty cells

//good values
//const int HASH_BITS=27;
//const int TURNS_LEFT=8;//hash only at least TURNS_LEFT empty cells

//Note USE_OWN_SET with big initial size slow down program
//This is uses for countAllNodes() and countAllNodes() functions and not used for end game solution
//#define USE_OWN_SET
//smaller position code size for countNodesFromStart and countNodesFromStartSymmetry 5x5
//#define CODE_5

#define NODE_COUNT


//Note #ifdef SHORT_GAME_SEARCH short shortest games in countNode() function, it's slow down countNode() function
//#define SHORT_GAME_SEARCH

//END basic defines & constants

#ifdef CODE_5
	#pragma pack(1) //structure size
	struct Code5{
		unsigned code;//should use unsigned!! see to64()
		unsigned short code1;//should use unsigned!! see to64()
		Code5(){code=code1=0;}
		Code5(uint64_t t){
			code = unsigned(t & 0xffffffff);
			code1 = (t>>32);
		}

		inline uint64_t to64()const{
			return (uint64_t(code1)<<32)|code;
		}

		inline bool operator<(const Code5& c)const{
			return code<c.code || (code==c.code && code1<c.code1);
		}
		inline bool operator==(const Code5& c)const{
			return code==c.code && code1==c.code1;
		}

		inline unsigned operator&(int v)const{
			return code&v;
		}

	};
	#define	SET_TYPE Code5
#else
	#define	SET_TYPE uint64_t
#endif

#ifdef USE_OWN_SET
	#include "Set.h"
	typedef Set<SET_TYPE> SetType;
#else
	#include <set>
	typedef std::set<SET_TYPE> SetType;
#endif

#define SIZE(a) int(sizeof(a)/sizeof((a)[0]))

const int END_ARRAY=0;
const char BLACK=0;
const char WHITE=1;
const char EMPTY=2;
const char OUT[]="BW-";
const int START_POSITION[][2]={ {1,2},{0,3},{2,3},{0,1},{0,2},{1,3} };

//for graphical interface to make universal interface for different Reversi objects
class ReversiBase{
public:
	virtual void set(int type)=0;/*type=0..5 START_POSITION[type]*/
	virtual void set(uint64_t code)=0;
	virtual void set(const char*p)=0;
	virtual void prepare()=0;
	virtual void free()=0;
	virtual int count(char c)const=0;
	virtual int getIndex(int x,int y)const=0;
	virtual bool possibleMove(int index,int white)const=0;
	virtual bool possibleMove(int white)const=0;//global possible move
	virtual char get(int x,int y)const=0;
	virtual char get(int index)const=0;
	virtual int estimateTurn(int index,int white,int alpha,int beta)const=0;
	virtual std::vector<int> flipList(int index,int white)const=0;
	virtual int makeMove(int index,int white)=0;
	virtual void getXY(int index,int&x,int&y)const=0;
	virtual std::string toString(std::string rowSeparator="\n")const=0;
	virtual std::vector<int> getOptimalMoves(int white,int alpha,int beta)const=0;
	virtual std::vector<int> getSymmetryCells(int index)const=0;
	virtual std::string toString(int index)const=0;
	virtual uint64_t hash(int type,int white)const=0;
	virtual uint64_t hash(int white)const=0;

	virtual int value(int alpha,int beta,int white)const=0;
	//search value using zero window search
	virtual int valueZeroWindow(int alpha,int beta,int white)const=0;

	virtual ~ReversiBase(){};

	static std::string uint64ToString(uint64_t v) {
		char c[128];
		_i64toa(v,c,10);

		int j=strlen(c);

		std::string s="";
		int i;
		for(i=0;i<j;i++){
			if(i%3==j%3 && i!=0){
				s+=',';
			}
			s+=c[i];
		}
		return s;
	}

	//solve positions from file
	virtual void solveFile(const char fileName[],int thread)=0;

};

//template is working much faster then using tableSize etc as class parameter,maximize of optimization criterion reversi/anti reversi
template<int tableSize,bool useHash,bool maximize=true> class Reversi : public ReversiBase{
public:
	static const int tableSize2=tableSize*tableSize;
	static const int fullTableSize=tableSize+2;
	static const int fullTableSize2=fullTableSize*fullTableSize;
	static const int maxDepth=tableSize2-4+2;
	static const int scoreInvalid=0xffffff;//can be >tableSize2 & <-tableSize2 because of cutoff
	static const char solvedMark='/';
	static const char solvingMark='+';
private:
	//data members at go first [static then non static]
	static TurnItem turnStart;
	static TurnItem turnEnd;
	static Reversi reversi[maxDepth];
	static Hash* hashTable;
	static const int HASH_SIZE=(1<<HASH_BITS);
	static const int HASH_AND_KEY=HASH_SIZE-1;
	static int minHashEmpty;
	//direction(i,j,k) = direction[((i)*9+j)*tableSize+k] Do not remove just for help
	static int direction[ (fullTableSize*(fullTableSize-1)-1) * 9 * tableSize ];
	static TurnItem turn[tableSize2];
	static int cellSymmetry[8][tableSize2];
#ifdef USE_FILE_TO_SOLVE
	static int minFileEmpty;
	static std::map<uint64_t,char> precountMap;
	static const int SOLVE_FILE_EMPTY = tableSize==5 ? 16 : 24;
#endif

public:
	static SetType nodesSet;
	static uint64_t nodeCount[maxDepth];//use for countNode function
#ifdef SHORT_GAME_SEARCH
	static std::string shortGameString[3];//use for search shortest games
	int currentTurn;
#endif
private:
	Reversi*next;
	int depth;
public:
	static void create();//common create
	char table[fullTableSize2];

	inline static Reversi& init(uint64_t code){
		Reversi r;
		r.set(code);
		return init(r);
	}
	inline static Reversi& init(const char*p){
		Reversi r;
		r.set(p);
		return init(r);
	}
	//type=0 classic, type=2
	static Reversi& init(int type,int moves,bool useSrand);
	static Reversi& init(Reversi const& r);
	
	static void destroy();//common destroy

	virtual void prepare(){
		create();
	}

	virtual void free(){
		destroy();
	}

	virtual int getIndex(int x,int y)const{
		return index(x,y);
	}

	inline void print()const{
		printf("%s\n",toString("\n").c_str());
	}

	virtual bool possibleMove(int white)const{
		int i;
		int*p=cellSymmetry[0];
		for(i=0;i<tableSize2;i++,p++){
			if(possibleMove(*p,white)){
				return true;
			}
		}
		return false;
	}

	virtual int estimateTurn(int index,int white,int alpha,int beta)const{
		Reversi r;
		r.copy(table);
		if(r.makeMove(index,white)==0){
			return scoreInvalid;
		}
		init(r);
		if(r.possibleMove(!white)){
			return -reversi->valueZeroWindow(-beta,-alpha,!white);
		}
		else{
			return  r.possibleMove(white) ?  reversi->valueZeroWindow(alpha,beta,white) : r.difference(white);
		}
	}

	virtual std::vector<int> getOptimalMoves(int white,int alpha,int beta)const{
		std::vector<int> v,v1,passed;
		int*p=cellSymmetry[0];
		int i,e;

		//check only one symmetry turn is possible, use 'v1', not use 'v', 'v' will be used later
		for(i=0;i<tableSize2;i++,p++){
			if(possibleMove(*p,white)){
				v1=getSymmetryCells(*p);
				break;
			}
		}
		for(;i<tableSize2;i++,p++){
			if(possibleMove(*p,white) && std::find(v1.begin(),v1.end(),*p)==v1.end() ){
				break;
			}
		}
		if(i==tableSize2){//one symmetry turn
			return v1;
		}

#ifdef USE_FILE_TO_SOLVE
		if( (tableSize==5 || tableSize==6) && count(EMPTY)==SOLVE_FILE_EMPTY){//we have precount estimate, so we can use fast zero window search
			e=difference(white)+precountMap[hash(white)];
			if(maximize){
				beta=e;
				alpha=beta-1;
			}
			else{
				alpha=e;
				beta=alpha+1;
			}
			for(p=cellSymmetry[0],i=0;i<tableSize2;i++,p++){
				if(possibleMove(*p,white) && std::find(passed.begin(),passed.end(),*p)==passed.end() ){
					e = estimateTurn(*p,white,alpha,beta);
					println("%d %d %d %d",alpha,beta,int(useHash),e);
					v1=getSymmetryCells(*p);
					passed.insert(passed.end(),v1.begin(),v1.end());
					if( (maximize && e>=beta) || (!maximize && e<=alpha) ){//found same estimation
						v.insert(v.end(),v1.begin(),v1.end());
						if(tableSize==6){//too slow count so find at least one turn
							break;
						}
					}
				}
			}
			assert(v.size()>0);
//			for(i=0;i<int(v.size());i++){
//				println("%s",toString(v[i]).c_str());
//			}
			return v;
		}
#endif

		for(p=cellSymmetry[0],i=0;i<tableSize2;i++,p++){
			if(possibleMove(*p,white) && std::find(passed.begin(),passed.end(),*p)==passed.end() ){
				e = maximize ? estimateTurn(*p,white,alpha-1,beta) : estimateTurn(*p,white,alpha,beta+1);
				v1=getSymmetryCells(*p);
				passed.insert(passed.end(),v1.begin(),v1.end());
				if( (maximize && e>=alpha) || (!maximize && e<=beta)){//found better estimate or same estimation
					if( (maximize && e>alpha) || (!maximize && e<beta) ){//better estimation
						if(maximize){
							alpha=e;
						}
						else{
							beta=e;
						}
						v.clear();
					}
					v.insert(v.end(),v1.begin(),v1.end());
				}
				if(alpha>=beta){
					return v;
				}
			}
		}
		return v;
	}

	/* for inner using. To get game estimation use value() function
	 * returns score changes
	 */
	int estimate(int alpha,int beta,int white)const;

	/* for inner using. To get game estimation use valueZeroWindow() function
	 * returns score changes
	 */
	int estimateZeroWindow(int alpha,int beta,int white)const;

	void copy(const char* from);

	void clear();

	virtual uint64_t hash(int type,int white)const;
	virtual uint64_t hash(int white)const;

	inline bool possibleMove(int index,int white)const{
		if(table[index]==EMPTY){
			return countScore(index,white,tableSize2)!=scoreInvalid;
		}
		else{
			return false;
		}
	}

	/**
	 * returns score
	 * result 0 means impossible move
	 */
	virtual int makeMove(int index,int white){
		assert(white==BLACK || white==WHITE);
		if(table[index]==EMPTY){
			int i=makeMoveInner(index,white);
			if(i!=0){//supports maximize & minimize
				table[index]=white;
			}
			return i;
		}
		else{
			return 0;
		}
	}

	inline char get(int x,int y)const{
		return table[index(x,y)];
	}

	inline char get(int index)const{
		return table[index];
	}

	inline void set(int x,int y,char c){
		table[index(x,y)]=c;
	}

	virtual void set(int type);
	virtual void set(uint64_t code);
	virtual void set(const char*p);

	virtual std::string toString(std::string rowSeparator="\n")const;

	void symmetry(int type);

	inline static int index(int x,int y){
		assert(x>=0 && x<tableSize);
		assert(y>=0 && y<tableSize);
		return (x+1)+(y+1)*fullTableSize;
	}

	virtual void getXY(int index, int& x,int& y)const{
		x=index%fullTableSize-1;
		y=index/fullTableSize-1;
		assert(x>=0 && x<tableSize);
		assert(y>=0 && y<tableSize);
	}

	/*
	 * result 0 means impossible move
	 */
	int makeMoveInner(int index,int white);

	/**
	 * returns score = number of flipped chips if maximize
	 * result=SCORE_INVALID means impossible move
	 */
	int countScore(int index,int white,int value)const;

	/*
	 * return true if at least one move can be found
	 */
	bool makeRandomMove(int white);

	int difference(int white)const;

	/*
	 * return position value
	 */
	virtual int value(int alpha,int beta,int white)const{
		int d=difference(white);
		return d+estimate(alpha-d,beta-d,white);
	}

	/*
	 * search value using zero window search
	 * return value is the same with value() function
	 * but this function works faster
	 */
	virtual int valueZeroWindow(int alpha,int beta,int white)const{
		int d=difference(white);
		return d+estimateZeroWindow(alpha-d,beta-d,white);
	}

	virtual int count(char what)const;

	virtual std::vector<int> flipList(int index,int white)const;

	//average time for position solution
	static void avgTime(int empties,int size,int type,int startPos,bool useSrand);

	//count onlyLastLayer=true is slower but use set only for last layer so can search deeper,
	//also this function search shortest games
	static void countAllNodes(int type,int minEmpties,bool useSymmetry);

	/**
	 * count nodes with only symmetry. This function is good for count all nodes of table 5x5, when we need to count many nodes
	 * to solve all 5x5 #define USE_OWN_SET and #define CODE_5
	 *
	 * if number of nodes on layer too big then need to call this function with totalClasses>4 parameter
	 * for example totalClasses=4 and we need to count on nodes on layer=19  (lastLayer=19)
	 * countAllNodesSymmetry(type, lastLayer=0, 4);
	 */
	static void countAllNodesSymmetry(int type, int lastLayer=0, int totalClasses=1);

	//Helper function for countAllNodes() & storeNodes()
	void countNodes(int white,int maxDepth,bool useSymmetry);
public:
	virtual std::vector<int> getSymmetryCells(int index)const{
		int i,j,k;
		std::vector<int> v;
		v.push_back(index);
		uint64_t c=hash(0,BLACK);
		for(j=0;j<tableSize2;j++){
			if(cellSymmetry[0][j]==index){
				break;
			}
		}
		assert(j!=tableSize2);
		for(i=1;i<SIZE(cellSymmetry);i++){
			if(hash(i,BLACK)==c){
				k=cellSymmetry[i][j];
				if(std::find(v.begin(),v.end(),k)==v.end()){//check that such item already in vector
					v.push_back(k);
				}
			}
		}
		return v;
	}

	virtual std::string toString(int index)const{
		int x,y;
		getXY(index,x,y);
		char c[3];
		c[0]='a'+x;
		c[1]='1'+y;
		c[2]=0;
		return c;
	}

	//BEGIN multithread solve file functions
	static void sleep(int milliseconds){
		clock_t begin=clock();
		while( (clock()-begin) < milliseconds*CLOCKS_PER_SEC/1000 );
	}

	static FILE* openSharedFile(const char* fileName){
		FILE*f;
		while( (f=_fsopen(fileName,"r+",_SH_DENYWR))==NULL && errno==EACCES){
			sleep(100);
		}
		if(f==NULL){
			printf("f==NULL error%d\n",errno);
		}
		return f;
	}

	//solve positions from file
	virtual void solveFile(const char fileName[],int thread);

	//output information about positions solution, numberIgnoreLines - ignore first 'numberIgnoreLines' lines
	static void solveFileInfo(const char fileName[],int numberIgnoreLines);

	//store nodes to file for further solution
	static void storeNodes(int minType,int maxType,int empties);

	//store nodes to file for further solution (store nodes which is in type=2 class, and not in type=0 class
	static void storeSNodes(int empties);

	//store set to i*.txt files
	static void storeSet(SetType const& set,int files);

	//transform file after solution
	static void transformFileAfterSolution(const char inFileName[],const char outFileName[]);

	//END multithread solve file function
};

template<int tableSize,bool useHash,bool maximize> TurnItem Reversi<tableSize,useHash,maximize>::turn[];
template<int tableSize,bool useHash,bool maximize> TurnItem Reversi<tableSize,useHash,maximize>::turnStart;
template<int tableSize,bool useHash,bool maximize> TurnItem Reversi<tableSize,useHash,maximize>::turnEnd;
template<int tableSize,bool useHash,bool maximize> int Reversi<tableSize,useHash,maximize>::cellSymmetry[][tableSize2];
#ifdef USE_FILE_TO_SOLVE
template<int tableSize,bool useHash,bool maximize> std::map<uint64_t,char> Reversi<tableSize,useHash,maximize>::precountMap;
template<int tableSize,bool useHash,bool maximize> int Reversi<tableSize,useHash,maximize>::minFileEmpty;
#endif
template<int tableSize,bool useHash,bool maximize> SetType Reversi<tableSize,useHash,maximize>::nodesSet;
template<int tableSize,bool useHash,bool maximize> int Reversi<tableSize,useHash,maximize>::direction[];
template<int tableSize,bool useHash,bool maximize> uint64_t Reversi<tableSize,useHash,maximize>::nodeCount[];
template<int tableSize,bool useHash,bool maximize> Reversi<tableSize,useHash,maximize> Reversi<tableSize,useHash,maximize>::reversi[];
template<int tableSize,bool useHash,bool maximize> Hash* Reversi<tableSize,useHash,maximize>::hashTable=NULL;
template<int tableSize,bool useHash,bool maximize> int Reversi<tableSize,useHash,maximize>::minHashEmpty;
#ifdef SHORT_GAME_SEARCH
template<int tableSize,bool useHash,bool maximize> std::string Reversi<tableSize,useHash,maximize>::shortGameString[];//use for search shortest games
#endif

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::set(int type) {
	assert(type>=0 && type<SIZE(START_POSITION));
	int j;
	const int i=tableSize/2-1;
	clear();
	const int* p=START_POSITION[type];
	for(j=0;j<4;j++){
		set(i+j%2,i+j/2, std::find(p,p+2,j)==p+2 ? BLACK : WHITE);
	}
}

template<int tableSize,bool useHash,bool maximize>
std::string Reversi<tableSize,useHash,maximize>::toString(std::string rowSeparator)const{
	int x,y;
	std::string s;
	for(y=0;y<tableSize;y++){
		for(x=0;x<tableSize;x++){
			s+=OUT[int(get(x,y))];
		}
		s+=rowSeparator;
	}
	return s;
}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::makeMoveInner(int index,int white){
	int score=0;
	int black=!white;
	const int *pi=direction+index*9*tableSize;
	const int *p;
	do{
		if(table[*pi]==black){
			for(p=pi+1 ; table[*p]==black ; p++ );
			if(table[*p]==white){
				p--;
				do{
					table[*p]=white;
					score++;
					p--;
				}while(p>=pi);
			}
		}
		pi+=tableSize;
	}while(*pi!=END_ARRAY);
	return score;
}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::countScore(int index, int white,int value) const {
	int score=1;
	int black=!white;
	const int *pi=direction+index*9*tableSize;
	const int *p;
	do{
		if(table[*pi]==black){
			for(p=pi+1 ; table[*p]==black ; p++);
			if(table[*p]==white){
				score+=(p-pi)<<1;
				if( score >= value ){
					return score;
				}
			}
		}
		pi+=tableSize;
	}while(*pi!=END_ARRAY);

	if(score==1){
		return scoreInvalid;
	}
	else{
		return score;
	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::set(const char*p){
	int i;
	int x,y;
	clear();
	for(y=0;y<tableSize;y++){
		for(x=0;x<tableSize;x++){
			while(*p==' '){
				p++;
			}
			assert(*p!=0);//too short string
			for(i=0;i<SIZE(OUT);i++){
				if(OUT[i]==*p){
					break;
				}
			}
			assert(i<SIZE(OUT));
			set(x,y,i);
			p++;
		}
	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::set(uint64_t code) {
	//Note this function should agree with hash() function
	int i;
	int*p=cellSymmetry[0];
	clear();
	code>>=1;
	if(tableSize%2==0){
		for(i=0;i<4;i++,p++){
			table[*p] = (code & 1) ? WHITE : BLACK;
			code>>=1;
		}

		p=cellSymmetry[0]+tableSize2-1;
		for(;i<tableSize2;i++,p--){
			table[*p] = code % 3;
			code/=3;
		}
	}
	else{
		p=cellSymmetry[0]+tableSize2-1;
		for(i=0;i<tableSize2;i++,p--){
			table[*p] = code % 3;
			code/=3;
		}

	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::copy(const char* from) {
	int i;
	char*p=table;
	const char*p1=from;
	for(i=0;i<fullTableSize2;i++){
		*p++=*p1++;
	}
}

template<int tableSize,bool useHash,bool maximize>
bool Reversi<tableSize,useHash,maximize>::makeRandomMove(int white) {
	int j,k,l;
	int i[tableSize2];
	int* p=i;
	int*c=cellSymmetry[0];
	char turn;
	Reversi r;
	r.clear();

	for(j=0;j<2;j++){
		turn = j==0 ?white : !white;

		for(k=0;k<tableSize2;k++){
			r.copy(table);
			l=*c++;
			if(r.makeMove(l,turn)){
				*p=l;
				p++;
			}
		}
		if(p!=i){
			makeMove(i[rand()%(p-i)],turn);
			return true;
		}
	}

	return false;

}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::difference(int white)const{
	int i,d=0;
	char c;
	int*p=cellSymmetry[0];
	const int black=!white;
	for(i=0;i<tableSize2;i++){
		c=table[*p++];
		if(c==white){
			d++;
		}
		else if(c==black){
			d--;
		}
	}
	return d;
}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::estimate(int alpha,int beta,int white)const{
#ifdef NODE_COUNT
	nodeCount[depth]++;
#endif
	int black=!white;
	int score,e,v;

#ifdef CODE_5
	printf("error estimate with CODE_5 defined\n");
	fflush(stdout);
	assert(0);
#endif

//	if(alpha<-tableSize2 ){//Using this code is error
//		alpha=-tableSize2;
//	}
//	if(beta>tableSize2){//Using this code is error
//		beta=tableSize2;
//	}
//	if(alpha>=beta){
//		return beta;
//	}

	uint64_t code=0;

#ifdef USE_FILE_TO_SOLVE
	if(tableSize==5 && depth==minFileEmpty){
		return precountMap[hash(white)];
	}
	else if(tableSize==6 && depth==minFileEmpty && maximize){
		code=hash(white);
		if(precountMap.find(code)!=precountMap.end()){
			return precountMap[code];
		}
	}
#endif

	Hash*pHash=0;
	char hashFlag=HASH_ALPHA;
	if(useHash && depth<=minHashEmpty){
		code=hash(0,white);//using of symmetry hash is slow down program
		pHash=hashTable+( (code^(code>>HASH_BITS)) & HASH_AND_KEY);
		if(pHash->getCode()==code){
			hashFlag=pHash->getFlag();
			if(hashFlag==HASH_EXACT){
				return pHash->value;
			}
			else if(hashFlag==HASH_ALPHA && alpha>=pHash->value){
				return alpha;
			}
			else if(hashFlag==HASH_BETA && beta<=pHash->value){
				return beta;
			}
		}
		hashFlag = maximize ? HASH_ALPHA : HASH_BETA;
	}

	TurnItem*t;
	t=turnStart.next;
	if(t->next==&turnEnd){
		score=countScore(t->turn,white,beta);
		if(score!=scoreInvalid){
			return score;
		}

		score=countScore(t->turn,black,-alpha);
		if(score!=scoreInvalid){
			return -score;
		}

		return 0;
	}

	bool found=false;

	next->copy(table);
	for(t=turnStart.next;t!=&turnEnd;t=t->next){
		score= next->makeMoveInner(t->turn,white);
		if(score){
			next->table[t->turn]=white;

			t->next->prev=t->prev;
			t->prev->next=t->next;

			v=(score<<1)|1;
			e=v-next->estimate( -beta+v,-alpha+v,black);
			t->prev->next=t->next->prev=t;

			if(maximize){
				if(e>alpha){
					alpha=e;
					if(alpha>=beta){
						if(pHash){
							pHash->set(code,HASH_BETA,beta);
						}
						return beta;
					}
					hashFlag=HASH_EXACT;
				}
			}
			else{
				if(e<beta){
					beta=e;
					if(alpha>=beta){
						if(pHash){
							pHash->set(code,HASH_ALPHA,alpha);
						}
						return alpha;
					}
					hashFlag=HASH_EXACT;
				}
			}

			found=true;
			next->copy(table);
		}
	}
	if(found){
		if(maximize){
			if(pHash){
				pHash->set(code,hashFlag,alpha);
			}
			return alpha;
		}
		else{
			if(pHash){
				pHash->set(code,hashFlag,beta);
			}
			return beta;
		}
	}

//	//Note this code is the same with recent code of this function. It's slower but useful for debug
//	if(possibleMove(black)){
//		return -estimate(-beta,-alpha,black);
//	}
//	else{
//		return 0;
//	}

	hashFlag = maximize ? HASH_BETA : HASH_ALPHA;
	for(t=turnStart.next;t!=&turnEnd;t=t->next){
		score= next->makeMoveInner(t->turn,black);
		if(score){
			next->table[t->turn]=black;

			t->next->prev=t->prev;
			t->prev->next=t->next;

			v=(score<<1)|1;
			e=-v+next->estimate( alpha+v,beta+v,white);
			t->prev->next=t->next->prev=t;

			if(maximize){
				if(e<beta){
					beta=e;
					if(alpha>=beta){
						if(pHash){
							pHash->set(code,HASH_ALPHA,alpha);
						}
						return alpha;
					}
					hashFlag=HASH_EXACT;
				}
			}
			else{
				if(e>alpha){
					alpha=e;
					if(alpha>=beta){
						if(pHash){
							pHash->set(code,HASH_BETA,beta);
						}
						return beta;
					}
					hashFlag=HASH_EXACT;
				}
			}
			found=true;
			next->copy(table);
		}
	}
	if(found){
		if(maximize){
			if(pHash){
				pHash->set(code,hashFlag,beta);
			}
			return beta;
		}
		else{
			if(pHash){
				pHash->set(code,hashFlag,alpha);
			}
			return alpha;
		}
	}
	else{
		return 0;
	}

}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::countNodes(int white,int maxDepth,bool useSymmetry){
	assert(next!=NULL);
	uint64_t code=hash(white);

	if(useSymmetry ){
		SetType& set=nodesSet;
		unsigned size=set.size();
#ifdef CODE_5
		Code5 code5(code);
		set.insert(code5);
#else
		set.insert(code);
#endif

		if(set.size()==size){
			return;
		}
		const int MN=10;
		if(size%(MN*1000000)==0 && size>0){
			time_t     now;
		  struct tm  *ts;
		  now = time(0);
		  ts = localtime(&now);
			char s[256];
		  strftime(s, 256, "%H:%M:%S", ts);
			printf("%2d meganodes %s\n",size/1000000,s);
			fflush(stdout);
		}

	}

	nodeCount[depth]++;

	if(depth>=maxDepth){
		//assert(turnStart.next==&turnEnd);
		return;
	}

	int black = !white;
	bool found=false;
	TurnItem*t;
	next->copy(table);
	for(t=turnStart.next;t!=&turnEnd;t=t->next){
		if(next->makeMove(t->turn,white)){
#ifdef SHORT_GAME_SEARCH
			currentTurn=t->turn;
#endif
			t->next->prev=t->prev;
			t->prev->next=t->next;
			next->countNodes(black,maxDepth,useSymmetry);
			t->prev->next=t->next->prev=t;
			next->copy(table);
			found=true;
		}
	}
	if(found){
		return;
	}

	for(t=turnStart.next;t!=&turnEnd;t=t->next){
		if(next->makeMove(t->turn,black)){
#ifdef SHORT_GAME_SEARCH
			currentTurn=t->turn;
#endif
			t->next->prev=t->prev;
			t->prev->next=t->next;
			next->countNodes(white,maxDepth,useSymmetry);
			t->prev->next=t->next->prev=t;
			next->copy(table);
			found=true;
		}
	}
#ifdef SHORT_GAME_SEARCH
	if(!found){
		int i;
		for(i=0;i<SIZE(shortGameString);i++){
			std::string& s=shortGameString[i];
			if(s.empty() || s.size()/2<=depth ){
				break;
			}
		}
		if(i==SIZE(shortGameString)){
			return;
		}

		bool hasBlack=false;
		bool hasWhite=false;
		int*p=cellSymmetry[0];
		for(i=0;i<tableSize2;i++,p++){
			if(table[*p]==BLACK){
				hasBlack=true;
			}
			else if(table[*p]==WHITE){
				hasWhite=true;
			}
		}

		if(hasBlack){
			if(hasWhite){
				i=EMPTY;
			}
			else{
				i=BLACK;
			}
		}
		else{
			assert(hasWhite);
			i=WHITE;
		}

		std::string& s=shortGameString[i];
		if( s.size()/2>depth || s.size()==0){
			s="";
			for(Reversi *p=reversi;p!=this;p++){
				s+=toString(p->currentTurn);
			}
		}

	}
#endif
}

template<int tableSize,bool useHash,bool maximize>
Reversi<tableSize,useHash,maximize>& Reversi<tableSize,useHash,maximize>::init(int type,int moves,bool useSrand) {
	int i;
	Reversi r;
	if(useSrand){
		srand (time(NULL));
	}
	do{
		r.set(type);
		for(i=0;i<moves;i++){
			if(!r.makeRandomMove( i%2==0 ? BLACK : WHITE )){
				//r.print();
				break;
			}
		}
		if(i<moves){
			printf("regenerate\n");
			fflush(stdout);
		}
	}while(i<moves);

	return init(r);
}

template<int tableSize,bool useHash,bool maximize>
Reversi<tableSize,useHash,maximize>& Reversi<tableSize,useHash,maximize>::init(Reversi<tableSize,useHash,maximize> const& r) {
	int i,j,x,y,k,x1,y1,*p,*p1;

	for(i=0;i<maxDepth;i++){
		reversi[i].next = i==SIZE(reversi)-1 ? NULL : reversi+i+1;
		reversi[i].depth=i;
		reversi[i].clear();
	}

#ifdef SHORT_GAME_SEARCH
	for(i=0;i<SIZE(shortGameString);i++){
		shortGameString[i]="";
	}
#endif

	for(i=0;i<maxDepth;i++){
		nodeCount[i]=0;
	}

	if(useHash && hashTable==NULL){
		hashTable=new Hash[HASH_SIZE];
		assert(hashTable!=NULL);
		for(i=0;i<HASH_SIZE;i++){
			hashTable[i].setInvalid();
		}
	}
	minHashEmpty=r.count(EMPTY)-TURNS_LEFT;
#ifdef USE_FILE_TO_SOLVE
	minFileEmpty=r.count(EMPTY)-SOLVE_FILE_EMPTY;
#endif

#ifdef NODE_COUNT
	for(i=0;i<maxDepth;i++){
		nodeCount[i]=0;
	}
#endif

	for(i=0;i<fullTableSize2;i++){
		reversi->table[i]=r.table[i];
	}

	int moveOrder[tableSize2];
	//set move order support tableSize 3-8
	assert(tableSize<=8);
	const int m[]={3,3, 0,0, 2,0, 2,2, 3,0, 3,2, 3,1, 2,1, 1,0, 1,1};

	j=0;
	for(i=0;i<SIZE(m);i+=2){
		if(maximize){
			x=m[i];
			y=m[i+1];
		}
		else{
			x=m[SIZE(m)-1-i];
			y=m[SIZE(m)-1-i-1];
		}
		if(x>=(tableSize+1)/2 || y>=(tableSize+1)/2){
			continue;
		}

		p=cellSymmetry[0];
		p1=std::find(p,p+tableSize2,index(x,y));
		assert(p1!=p+tableSize2);

		x1=p1-p;
		for(k=0;k<8;k++){
			y1=cellSymmetry[k][x1];
			if(std::find(moveOrder,moveOrder+j,y1)==moveOrder+j){
				assert(j<tableSize2);
				moveOrder[j++]=y1;
			}
		}

	}

	assert(j==tableSize2);
#ifndef NDEBUG
	//check all cells are presented
	p=cellSymmetry[0];
	p1=moveOrder+tableSize2;
	for(i=0;i<tableSize2;i++,p++){
		assert(std::find(moveOrder,p1,*p)!=p1);
	}
#endif

	//turn list
	for(i=0;i<tableSize2;i++){
		turn[i].next = i==SIZE(turn)-1  ? &turnEnd : turn+i+1;
		turn[i].prev = i==0  ? &turnStart : turn+i-1;
	}
	turnStart.next=turn;

	p=moveOrder;
	TurnItem*pTurn=turn;
	for(int i=0;i<tableSize2;i++,p++){
		if(reversi->table[*p]==EMPTY){
			pTurn->turn=*p;
			pTurn++;
		}

	}
	(--pTurn)->next=&turnEnd;

	return reversi[0];
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::clear() {
	for(int i=0;i<fullTableSize2;i++){
		table[i]=EMPTY;
	}
}

template<int tableSize,bool useHash,bool maximize>
uint64_t Reversi<tableSize,useHash,maximize>::hash(int white) const {
	int i;
	uint64_t codeMin=hash(0,white);
	uint64_t code;
	for(i=1;i<8;i++){
		code=hash(i,white);
		if(code<codeMin){
			codeMin=code;
		}
	}
	return codeMin;
//	if(tableSize%2==0){
//		for(i=1;i<8;i++){
//			code=hash(i,white);
//			if(code<codeMin){
//				codeMin=code;
//			}
//		}
//		return codeMin;
//	}
//	else{
//		int j;
//
//		for(i=1;i<8;i++){
//			for(j=0;j<4;j++){
//				if(table[cellSymmetry[i][j]]==EMPTY){
//					break;
//				}
//			}
//			if(j==4){
//				code=hash(i,white);
//				if(code<codeMin){
//					codeMin=code;
//				}
//			}
//		}
//		return codeMin;
//	}
}

template<int tableSize,bool useHash,bool maximize>
uint64_t Reversi<tableSize,useHash,maximize>::hash(int type,int white) const {
	int i;
	uint64_t code=0;
	int*p=cellSymmetry[type];
	if(tableSize%2==0){
		for(i=0;i<4;i++,p++){
			if(table[*p]==WHITE){
				code|=(1<<i);
			}
		}
		code<<=1;
		code|=white;

		uint64_t code1=0;
		for(;i<tableSize2;i++,p++){
			code1=(code1<<1)+code1;//code1*=3;
			code1+=table[*p];
		}

		return (code1<<5)|code;
	}
	else{
		for(i=0;i<tableSize2;i++,p++){
			code=(code<<1)+code;//code*=3;
			code+=table[*p];
		}
		return (code<<1)|white;
	}
}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::estimateZeroWindow(int alpha, int beta, int white)const {
	int i,v;

	while(alpha+1!=beta){
		i=(alpha+beta)/2;
		v=estimate(i,i+1,white);
		if(v>=i+1){
			if(alpha==v){
				return alpha;
			}
			alpha=v;
		}
		else{
			if(beta==v){
				return beta;
			}
			beta=v;
		}
		if(alpha==beta){
			return alpha;
		}
	}
	return estimate(alpha,beta,white)<=alpha ? alpha : beta;
}

template<int tableSize,bool useHash,bool maximize>
int Reversi<tableSize,useHash,maximize>::count(char what) const {
	int i;
	int c=0;
	int*p=cellSymmetry[0];
	for(i=0;i<tableSize2;i++){
		if(table[*p++]==what){
			c++;
		}
	}
	return c;
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::symmetry(int type) {
	Reversi r;
	assert(type>=0 && type<8);
	r.copy(table);

	int i;
	int*p=cellSymmetry[type];
	int*p1=cellSymmetry[0];
	for(i=0;i<tableSize2;i++){
		table[*p1++]=r.table[*p++];
	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::create() {
	int i,j,k,ii,x,y,x1,y1,*p;
	//set directions
	const int add[8][2]={	{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,-1},{-1,1},{1,-1} };
	for(y=0;y<tableSize;y++){
		for(x=0;x<tableSize;x++){
			ii=index(x,y);
			j=0;
			for(i=0;i<8;i++){
				x1=x+2*add[i][0];
				y1=y+2*add[i][1];
				if(x1>=0 && y1>=0 && x1<tableSize && y1<tableSize){
					x1=x+add[i][0];
					y1=y+add[i][1];
					for( p=direction+(ii*9+j)*tableSize ; x1>=0 && y1>=0 && x1<tableSize && y1<tableSize ; x1+=add[i][0],y1+=add[i][1],p++ ){
						*p=index(x1,y1);
					}
					*p=END_ARRAY;
					j++;
				}
			}
			direction[(ii*9+j+1)*tableSize]=END_ARRAY;
		}
	}

	//set cell symmetry, first four items are initial items
	i=tableSize/2-1;
	int a[][2]={ {i,i},{i+1,i+1},{i,i+1},{i+1,i} };
	int b[tableSize2][2];

	for(j=0;j<4;j++){
		b[j][0]=a[j][0];
		b[j][1]=a[j][1];
	}

	for(y=0;y<tableSize;y++){
		for(x=0;x<tableSize;x++){
			for(k=0;k<4;k++){
				if(a[k][0]==x && a[k][1]==y){
					break;
				}
			}
			if(k==4){
				b[j][0]=x;
				b[j][1]=y;
				j++;
			}
		}
	}

	for(i=0;i<8;i++){
		p=cellSymmetry[i];
		for(j=0;j<tableSize2;j++){
			x=b[j][0];
			y=b[j][1];
			if(i & 1){
				x1=y;
				y1=x;
			}
			else{
				x1=x;
				y1=y;
			}
			if(i & 2){
				x1=tableSize-1-x1;
			}
			if(i & 4){
				y1=tableSize-1-y1;
			}
			*p++=index(x1,y1);
		}

	}

#ifdef USE_FILE_TO_SOLVE
	if(tableSize==5 || tableSize==6){
		SetType full;
		Reversi re;
		SetType::const_iterator it;

		for(i=0;i<SIZE(START_POSITION);i++){
			Reversi&r=init(i,0,false);
			r.countNodes(BLACK,tableSize2-4-SOLVE_FILE_EMPTY,true);
			SetType& set=nodesSet;
			for(it=set.begin();it!=set.end();it++){
				re.set(*it);
				if(re.count(EMPTY)==SOLVE_FILE_EMPTY){
					full.insert(*it);
				}
			}

			if(tableSize==6){
				break;
			}

		}
		char c[20];
		sprintf(c,"%d%s%d.dat",tableSize,maximize ? "max":"min",SOLVE_FILE_EMPTY);
		FILE*f=fopen(c,"rb");
		assert(f!=NULL && "USE_FILE_TO_SOLVE not found");
		for(it=full.begin();it!=full.end();it++){
			fread(c,1,1,f);
			precountMap[*it]=c[0];
		}
		fclose(f);
	}
#endif
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::avgTime(int empties, int size, int type, int startPos,bool useSrand) {
	int i;
	printf("empties=%d SIZE=%d table=%dx%d\n",empties,size,tableSize,tableSize);
	fflush(stdout);
	clock_t beginProgram=clock();
	clock_t t,at;
	for(i=0;i<size;i++){
		Reversi& r=init(startPos,tableSize2-4-empties,useSrand);

		if(type==0){
			r.value(0,1,WHITE);
		}
		else if(type==1){
			r.valueZeroWindow(-tableSize2,tableSize2,WHITE);
		}
		else{
			r.value(-tableSize2,tableSize2,WHITE);
		}

		t=clock()-beginProgram;
		at=t/(i+1);
		printf("%2d total %02ld:%02ld.%03ld avg %02ld:%02ld.%03ld\n",i
				,t/CLOCKS_PER_SEC/60,t/CLOCKS_PER_SEC%60, t%CLOCKS_PER_SEC
				,at/CLOCKS_PER_SEC/60,at/CLOCKS_PER_SEC%60, at%CLOCKS_PER_SEC
				);
		fflush(stdout);
	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::countAllNodesSymmetry(int type, int lastLayer, int totalClasses) {
	clock_t t,begin=clock(),beginLayer;
	SetType::const_iterator it;
	int color=BLACK;
	int i,layer,*p,ps;
	Reversi r,r1;
	bool found;
	uint64_t code,c1,c2;
	uint64_t total=0;
	int currentClass;
	FILE*f;
#if defined(USE_OWN_SET)
	const int MN=tableSize==5 ?
#ifdef CODE_5
			130
#else
			100
#endif
 : 130;
	nodesSet.init(MN*1000000);
#endif

#ifdef CODE_5
	Code5 c5;
#endif

	printf("countAllNodesSymmetry table%dx%d type%d code5:%s useOwnSet:%s lastLayer%d totalClasses%d",tableSize,tableSize,type,
#ifdef CODE_5
			"on"
#else
			"off"
#endif
			,
#ifdef USE_OWN_SET
			"on"
#else
			"off"
#endif

			,lastLayer,totalClasses
			);

#if defined(USE_OWN_SET)
	printf(" setSize%dmn",MN);
#endif

	printf("\n");
	fflush(stdout);

#ifdef CODE_5
	if(tableSize>5){
		printf("countNodesFromStartSymmetry error CODE_5 defined and tableSize>5\n");
		fflush(stdout);
		return;
	}
#endif

#ifdef CODE_5
	#define RW c5
#else
	#define RW code
#endif

	r.set(type);
	nodesSet.insert(r.hash(BLACK));
	total++;

	printf("%2d %14s bf=1.00 time 00:00:00",tableSize2-4,"1" );
	fflush(stdout);

	for(layer=tableSize2-4-1;layer>=lastLayer;layer--){
		ps=nodesSet.size();
		beginLayer=clock();

		//store set to file
		f=fopen("snodes.dat","wb+");
		for(it=nodesSet.begin();it!=nodesSet.end();it++){
			RW=*it;
			fwrite(&RW,sizeof(SET_TYPE),1,f);
		}
		for( currentClass=0 ; currentClass<(layer==lastLayer ? totalClasses : 1) ; currentClass++){
			nodesSet.clear();
			fseek(f,0,SEEK_SET);
			fflush(f);
			while(fread(&RW,sizeof(SET_TYPE),1,f)==1){
#ifdef CODE_5
				code=c5.to64();
#endif
				color=RW&1;

				found=false;
				p=cellSymmetry[0]+4;
				for(i=0;i<tableSize2;i++,p++){
					r1.set(code);
					if(r1.makeMove(*p,color)!=0){
						found=true;
						c1=r1.hash(!color);
						c2=((c1>>3)^(c1>>6)^(c1>>9)^(c1>>12))%totalClasses;
						if( (layer==lastLayer && c2==currentClass) || layer>lastLayer ){
							nodesSet.insert(c1);
						}
						r1.set(code);
					}
				}
				if(!found){
					p=cellSymmetry[0]+4;
					for(i=0;i<tableSize2;i++,p++){
						if(r1.makeMove(*p,!color)!=0){
							c1=r1.hash(color);
							c2=((c1>>3)^(c1>>6)^(c1>>9)^(c1>>12))%totalClasses;
							if( (layer==lastLayer && c2==currentClass) || layer>lastLayer ) {
								nodesSet.insert(c1);
							}
							r1.set(code);
						}
					}
				}
			}
			t=(clock() - begin)/CLOCKS_PER_SEC;
			printf("\n%2d %14s bf=%.2lf time %02ld:%02ld:%02ld",layer,uint64ToString(nodesSet.size()).c_str(),double(nodesSet.size())/ps
					,t/60/60, (t/60)%60,t%60 );

			t=(clock() - beginLayer)/CLOCKS_PER_SEC;
			printf(" %s time %02ld:%02ld:%02ld",totalClasses!=1 && layer==lastLayer ? "layer": "class",t/60/60, (t/60)%60,t%60 );
			if(totalClasses!=1 && layer==lastLayer ){
				printf(" class %d/%d",currentClass+1,totalClasses );
			}
			fflush(stdout);
			total+=nodesSet.size();
		}
		fclose(f);

	}

	t=(clock() - begin)/CLOCKS_PER_SEC;
	printf("\n** %14s time %02ld:%02ld:%02ld",uint64ToString(total).c_str(),t/60/60, (t/60)%60,t%60 );
	fflush(stdout);

}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::countAllNodes(int type,int minEmpties,bool useSymmetry) {
	int i,e,total;
	clock_t t,beginProgram=clock();
	Reversi&r=init(type,0,false);
	printf("countAllNodes table%dx%d type%d symmetry%d minEmpties%d\n",tableSize,tableSize,type,useSymmetry?1:0,minEmpties);
	fflush(stdout);

	if(useSymmetry ){
#if defined(USE_OWN_SET)
		if(tableSize==5){
			nodesSet.init( 108418253);
		}
		else{
			nodesSet.init( 108418253);
		}
#else
		nodesSet.clear();
#endif
	}

	e=tableSize2-4-minEmpties;
	r.countNodes(BLACK,e,useSymmetry);
	for(total=i=0;i<=e;i++){
		printf("%2d %14s bf=%.2lf\n",r.count(EMPTY)-i,uint64ToString(nodeCount[i]).c_str()
				, i==0 ? 1. : double(nodeCount[i])/(nodeCount[i-1]) 		);
		total+=nodeCount[i];
	}
#ifdef SHORT_GAME_SEARCH
	printf("short game turns ");
	for(i=0;i<3;i++){
		printf("%c%d %s",i==0?'[':' ',shortGameString[i].size()/2,shortGameString[i].c_str());
	}
	printf("]\n");
#endif
	t=(clock() - beginProgram)/CLOCKS_PER_SEC;
	printf("total time %02ld:%02ld total nodes=%s\n\n",t/60,t%60,uint64ToString(total).c_str() );
	fflush(stdout);
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::storeNodes(int minType,int maxType,int empties) {
#if defined(USE_OWN_SET) || defined(CODE_5)
	printf("storeNodes doesn't support USE_OWN_SET or CODE_5\n");
#else
	assert(minType<=maxType);
	int type,total,insert;
	Reversi re;
	unsigned sz;
	SetType full;
	for(type=minType;type<=maxType;type++){
		Reversi&r=init(type,0,false);
		SetType::const_iterator it;
		SetType& set=nodesSet;
		set.clear();
		r.countNodes(BLACK,tableSize2-4-empties,true);

		insert=0;
		total=0;
		for(it=set.begin();it!=set.end();it++){
			re.set(*it);
			if(re.count(EMPTY)==empties){
				sz=full.size();
				full.insert(*it);
				if(full.size()!=sz){
					insert++;
				}
				total++;
			}
		}
		printf("type%d inserted %6d/%6d %6.2lf%% total %6d\n",type,insert,total,100.*insert/total,full.size());
	}
	storeSet(full,1);
#endif
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::storeSNodes(int empties) {
#if defined(USE_OWN_SET) || defined(CODE_5)
	printf("storeNodes doesn't support USE_OWN_SET or CODE_5\n");
#else
	int type,total,insert;
	Reversi re;
	unsigned sz;
	SetType full,store;
	for(type=0;type<=2;type+=2){
		Reversi&r=init(type,0,false);
		SetType::const_iterator it;
		SetType& set=nodesSet;
		set.clear();
		r.countNodes(BLACK,tableSize2-4-empties,true);

		insert=0;
		total=0;
		for(it=set.begin();it!=set.end();it++){
			re.set(*it);
			if(re.count(EMPTY)==empties){
				sz=full.size();
				full.insert(*it);
				if(full.size()!=sz){
					insert++;
					if(type==2){
						store.insert(*it);
					}
				}
				total++;
			}
		}
		printf("type%d inserted %6d/%6d %6.2lf%% total %6d\n",type,insert,total,100.*insert/total,full.size());
	}
	storeSet(store,1);
#endif
}


template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::solveFile(const char fileName[],int thread) {
	FILE*f;
	uint64_t code,newCode;
	const int BUFF_SIZE=128;
	char c[BUFF_SIZE];
	clock_t begin,t;
	int e;
	bool bwrite;

	printf("start thread%d\n",thread);
	fflush(stdout);

	//Note when read string from solving file now just one digit is used
	assert(thread>=0 && thread<10);

	code=0;
	t=0;

	while(1) {

		bwrite=false;
		newCode=0;

		f=openSharedFile(fileName);
		while(fgets(c,BUFF_SIZE,f)){
			if(c[1]==solvingMark && c[0]=='0'+thread){
				if(code==0){//only read code
					newCode=_atoi64(c+2);
					break;
				}
				else{
					assert(uint64_t(_atoi64(c+2))==code);
					fseek(f,ftell(f)-strlen(c)-1,SEEK_SET);
					//use short string
					fprintf(f,"%c%s %d %ld:%02ld:%02ld t%d %s %d",solvedMark,_i64toa(code,c,10),e,t/3600,(t/60)%60,t%60
							,thread,maximize ?"max":"min",tableSize);
					fflush(f);
					bwrite=true;
					if(newCode!=0){
						break;
					}
				}
			}
			else if(newCode==0 && isdigit(c[0]) && isdigit(c[1])){//skip '0+code' and '/code'
				newCode=_atoi64(c);
				fseek(f,ftell(f)-strlen(c)-1,SEEK_SET);
				fprintf(f,"%d%c%s",thread,solvingMark,_i64toa(newCode,c,10));//have to use _i64toa here because we don't want to write whole string 'c'
				fflush(f);
				if(bwrite){
					break;
				}
			}
		}
		fclose( f );

		if(code!=0){
			assert(bwrite);
		}

		if(newCode==0){
			break;//no new codes
		}
		code=newCode;

//		printf("thread%d %s\n", thread,_i64toa(code,c,10));
//		fflush(stdout);

		begin=clock();

		Reversi& r=init(code);
		e=r.valueZeroWindow(-tableSize2,tableSize2,code&1);

		t=(clock()-begin)/CLOCKS_PER_SEC;
//		printf("solved code thread%d %s %d\n",thread,_i64toa(code,c,10),e);
//		fflush(stdout);

	}


	printf("thread%d finished\n",thread);
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::destroy() {
	if(useHash && hashTable){
		delete[]hashTable;
		//DO not remove, in graphical interface user can select game type with hash:on then switch to another game type
		//, then return to this game type so need to allocate hash second time
		hashTable=NULL;
	}
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::storeSet(const SetType& set, int files) {
#if defined(USE_OWN_SET) || defined(CODE_5)
	printf("storeSet doesn't support USE_OWN_SET or CODE_5\n");
#else
	const int n=set.size();
	int i,j,k;
	char c[64];
	FILE*f;
	SetType::const_iterator it=set.begin();
	for(k=i=0;i<files;i++){
		if(files==1){
			sprintf(c,"i.txt");
		}
		else{
			sprintf(c,"i%d.txt",i);
		}
		f=fopen(c,"w+");
		for(j=0;j<n/files+(i<n%files);j++,k++,it++){
			fprintf(f,"%s%30s\n",_i64toa(*it,c,10)," ");
		}
		fclose(f);
		printf("file%d items%d\n",i,j);
	}
	printf("total %d\n",k);
#endif
}

template<int tableSize,bool useHash,bool maximize>
std::vector<int> Reversi<tableSize,useHash,maximize>::flipList(int index, int white) const {
	std::vector<int> v;
	Reversi r;
	r.copy(table);
	r.makeMove(index,white);
	for(int i=0;i<fullTableSize2;i++){
		if(table[i]!=EMPTY && table[i]!=r.table[i]){
			v.push_back(i);
		}
	}
	return v;
}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::solveFileInfo(const char fileName[],int numberIgnoreLines) {
	int solved=0,total=0;
	int minTime=INT_MAX,maxTime=INT_MIN,totalTime=0;
	int h,m,s,n;
	const int BUFF_SIZE=128;
	char c[BUFF_SIZE];
	int threads=-1;
	char*p;
	FILE*f=openSharedFile(fileName);
	if(f==NULL){
		printf("error cann't open file %s",fileName);
		return;
	}
	for(n=0;fgets(c,BUFF_SIZE,f);n++){
		if(n<numberIgnoreLines){
			continue;
		}
		if(c[0]==solvedMark){
			p=strchr(c,':');
			assert(p!=NULL);
			while(*p!=' ' && p!=c){
				p--;
			}
			p++;
			sscanf(p,"%d:%d:%d",&h,&m,&s);
			h=s+m*60+h*3600;
			totalTime+=h;
			if(minTime>h){
				minTime=h;
			}
			if(maxTime<h){
				maxTime=h;
			}

			p+=6;
			p=strchr(p,'t');
			if(p!=NULL){//old file format has no thread mark 't0'
				h=atoi(p+1)+1;
				if(h>threads){
					threads=h;
				}
			}
			solved++;
		}
		total++;
	}
	fclose(f);

	printf("solved %d/%d %.2lf%% threadsFound %d\n",solved,total,solved*100./total,threads);
	if(solved==0){
		return;
	}
	h=totalTime/solved;
	m=h*(total-solved)/threads;
	printf("time min %02d:%02d:%02d max %02d:%02d:%02d total %02d:%02d:%02d avg %02d:%02d:%02d left %02d:%02d:%02d %.2lf days\n"
			,minTime/3600,(minTime/60)%60,minTime%60
			,maxTime/3600,(maxTime/60)%60,maxTime%60
			,totalTime/3600,(totalTime/60)%60,totalTime%60
			,h/3600,(h/60)%60,h%60
			,m/3600,(m/60)%60,m%60
			,m/3600./24.
	);
	printf("Note. Average and left time are counted incorrectly if processes were suspended\n");
	fflush(stdout);

}

template<int tableSize,bool useHash,bool maximize>
void Reversi<tableSize,useHash,maximize>::transformFileAfterSolution(const char inFileName[],const char outFileName[]){
	FILE*in=fopen(inFileName,"r");
	FILE*out=fopen(outFileName,"wb+");
	assert(in!=NULL);
	assert(out!=NULL);

	const int BUFF_SIZE=128;
	char c[BUFF_SIZE];
	char*p;
	uint64_t code;
	Reversi r;
	int value,e;

	while(fgets(c,BUFF_SIZE,in)){
		assert(c[0]==solvedMark);
		code=_atoi64(c+1);
		p=strchr(c,' ');
		assert(p!=NULL);
		value=atoi(p);

		r.set(code);

		e=value-r.difference(code&1);
		assert( e>=SCHAR_MIN && e<=SCHAR_MAX );
		c[0]=e;
		fwrite(c,1,1,out);
	}

	fclose(in);
	fclose(out);
}


#endif /* REVERSI_H_ */
