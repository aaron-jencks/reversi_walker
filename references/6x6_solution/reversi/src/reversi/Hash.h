/*
 * Hash.h
 *
 *  Created on: 24.01.2015
 *      Author: alexey slovesnov
 */

#ifndef HASH_H_
#define HASH_H_

#include <stdint.h>

#pragma pack(1) //struct size
static const char HASH_ALPHA=0;
static const char HASH_BETA =1;
static const char HASH_EXACT=2;
static const char HASH_INVALID=3;

class Hash {
public:

	uint64_t codeFlag;
	char value;
	char depth;

	void set(uint64_t code,char flag,char _value){
		codeFlag=(code<<2)|flag;
		value=_value;
	}

	inline uint64_t getCode()const{
		return codeFlag>>2;
	}

	inline char getFlag()const{
		return codeFlag&3;
	}

	inline void setInvalid(){
		codeFlag=HASH_INVALID;
	}

};

#endif /* HASH_H_ */
