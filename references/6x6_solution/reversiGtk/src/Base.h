/*
 * Base.h
 *
 *  Created on: 28.01.2015
 *      Author: alexey slovesnov
 */

#ifndef BASE_H_
#define BASE_H_

#include <gtk/gtk.h>
#include <string>
#include <string.h>
#include <assert.h>

const GdkRGBA blackColor= {0.,0.,0.,1.};
const GdkRGBA whiteColor= {1.,1.,1.,1.};
const GdkRGBA bestColor= {155/255., 0/255., 0/255. ,1.};

static const char HOMEPAGE[]="http://slovesnov.users.sourceforge.net?reversi_small";
static const int MINIMAL_TABLE_SIZE=3;
static const char CONFIG_FILENAME[]="reversi.cfg";

#ifndef NDEBUG
//just for debug output file,line,function and what was set and goes to new line, using like 'println("%d %d",i,j)'
//note use "__" precedence to prevent potential conflict with inner variables of functions
#define println(f, ...)  {\
	char __s[1024],__b[127];\
	sprintf(__b,__FILE__);\
	char*__p=strrchr(__b,G_DIR_SEPARATOR);\
	sprintf(__s,f,##__VA_ARGS__);\
	g_print("%-40s %s:%d %s()\n",__s,__p==NULL?__b:__p+1,__LINE__,__func__);\
}

//just for debug output just file,line,function
//Note not use 'print' instead of 'printinfo' because it's already defined in include from <sstream>
#define printinfo println(" ")

#else

#define println(f, ...)  ((void)0)
#define printinfo

#endif

//after #define println
#define GTK_REVERSI
#include "../../reversi/src/reversi/Reversi.h"

class Base {
	static GdkPixbuf* m_cellPixbuf[3];
public:
	static GdkPixbuf * m_background;
	static GdkPixbuf * m_possiblePixbuf;

  static int m_cellSize;

  static const int m_fontSize=18;
  static const int m_turnFontSize=13;

	Base();
	virtual ~Base();

	static inline GdkPixbuf* chip(int color){
		assert(BLACK==0 && WHITE==1 && EMPTY==2);
		assert(color>=0 && color<SIZE(m_cellPixbuf) );
		return m_cellPixbuf[color];
	}

  static inline std::string getImagePath(const char img[]){
  	return std::string("img/")+img;
  }

	template <class T>static inline void createNew(T*& dest, T* source){
		destroy(dest);
		dest=source;
	}

	static inline void destroy(cairo_t* p){
		if(p!=NULL){
			cairo_destroy(p);
		}
	}

	static inline void destroy(cairo_surface_t * p){
		if(p!=NULL){
			cairo_surface_destroy(p);
		}
	}

	static inline void copy(cairo_surface_t * source,cairo_t * dest,int destx, int desty
			,int width, int height, int sourcex, int sourcey){
		cairo_set_source_surface (dest, source, destx-sourcex, desty-sourcey);
		cairo_rectangle (dest, destx, desty, width, height);
		cairo_fill (dest);
	}

	static inline void copy(GdkPixbuf* source,cairo_t * dest,int destx, int desty){
		const int sourcex=0;
		const int sourcey=0;
		int width=gdk_pixbuf_get_width(source);
		int height=gdk_pixbuf_get_height(source);
		gdk_cairo_set_source_pixbuf (dest,source,destx-sourcex,desty-sourcey);
		cairo_rectangle(dest,destx,desty, width,height);
		cairo_fill(dest);
	}

	static inline GdkPixbuf* pixbuf(const char* _format, ...){
	  char buffer[256];
	  va_list args;
	  va_start (args, _format);
	  vsprintf (buffer,_format, args);
	  va_end (args);
		return gdk_pixbuf_new_from_file(getImagePath(buffer).c_str(), NULL);
	}

	static inline GtkWidget* image(const char* n){
		return gtk_image_new_from_file(getImagePath(n).c_str());
	}

	static inline const gchar * utf8ToLocale(std::string s){
		return g_locale_from_utf8(s.c_str(),s.length(),NULL,NULL,NULL);
	}


};

#endif /* BASE_H_ */
