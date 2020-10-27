/*
 * Base.cpp
 *
 *  Created on: 28.01.2015
 *      Author: alexey slovesnov
 */

#include "Base.h"

int Base::m_cellSize;
GdkPixbuf * Base::m_background=NULL;
GdkPixbuf* Base::m_cellPixbuf[];
GdkPixbuf * Base::m_possiblePixbuf=NULL;

Base::Base() {
	int i;
	if(!m_background){
		for(i=0;i<SIZE(m_cellPixbuf);i++){
			m_cellPixbuf[i] = pixbuf ("r%d.png",i);
		}
		m_background=pixbuf("bg0.jpg");
		m_possiblePixbuf=pixbuf("possible.png");
		m_cellSize = gdk_pixbuf_get_height(m_cellPixbuf[0]);
	}
}

Base::~Base() {
}
