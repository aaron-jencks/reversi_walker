/*
 * Set.h
 *
 *	std::set is not working with big arrays so should use my own
 *  Created on: 15.01.2015
 *      Author: alexey slovesnov
 */

#ifndef SET_H_
#define SET_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

template<typename type> class Set {
	static const int SET_BITS=26;
	static const int START_SET_SIZE=(1<<SET_BITS);

	struct Item{
		type code;
		Item* next;
	};

	Item* set;
	Item** start;
	unsigned m_size;
	int max_size;
public:
	class iterator{
		Item*ptr;
	public:
		iterator(Item *p=NULL){
			ptr=p;
		}

		inline bool operator!=(iterator it)const{
			return ptr!=it.ptr;
		}

		inline bool operator==(iterator it)const{
			return ptr==it.ptr;
		}

		inline type operator*()const{
			return ptr->code;
		}

		//postfix ++ (works as prefix)
		inline iterator& operator++(int){
			ptr++;
			return *this;
		}
	};
	typedef iterator const_iterator;

	Set(){
		start=NULL;
		set=NULL;
		m_size=0;
		max_size=0;
	}

	virtual ~Set(){
		deinit();
	}

	void clear(){
		m_size=0;
		for(int i=0;i<START_SET_SIZE;i++){
			start[i]=NULL;
		}
	}

	void init(int size){
		m_size=0;
		deinit();
		max_size=size;
		set=new Item[size];
		assert(set!=NULL);
		start=new Item*[START_SET_SIZE];
		assert(start!=NULL);
		for(int i=0;i<START_SET_SIZE;i++){
			start[i]=NULL;
		}
	}

	void deinit(){
		if(set!=NULL){
			delete[]set;
		}
		if(start!=NULL){
			delete[]start;
		}
	}

	inline unsigned size()const{
		return m_size;
	}

	inline iterator begin()const{
		return iterator(set);
	}

	inline iterator end()const{
		return iterator(set+m_size);
	}

	inline iterator find(type t)const{
		Item * p=start [t & (START_SET_SIZE-1) ];
		if(p==NULL){
			return end();
		}
		else{
			for(;p->next!=NULL;p=p->next){
				if(p->code==t){
					return p;
				}
			}
			if(p->code==t){
				return p;
			}
			return end();
		}
	}


	void insert(type t){
		Item * p=start [t & (START_SET_SIZE-1) ];
		if(p==NULL){
			p=start [t & (START_SET_SIZE-1) ]=set+m_size;
		}
		else{
			for(;p->next!=NULL;p=p->next){
				if(p->code==t){
					return;
				}
			}
			if(p->code==t){
				return;
			}
			p=p->next=set+m_size;
		}
		assert(m_size<max_size);//this assert is checked and valid
		p->code=t;
		p->next=0;
		m_size++;
	}
};

#endif /* SET_H_ */
