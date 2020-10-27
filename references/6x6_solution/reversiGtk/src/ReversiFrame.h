/*
 * ReversiFrame.h
 *
 *  Created on: 25.01.2015
 *      Author: alexey slovesnov
 */

#ifndef REVERSIFRAME_H_
#define REVERSIFRAME_H_

#include "Base.h"
#include "ReversiArea.h"
#include "ScoreTurnArea.h"

enum {
	SHOW_TYPE_POSSIBLE_MOVES,
	SHOW_TYPE_ESTIMATES,
	SHOW_TYPE_NOTHING
};

enum {
	COMBO_TABLE_SIZE,
	COMBO_SHOW_OPTION,
	COMBO_PLAYER,
	COMBO_START_POSITION,

	COMBO_SIZE
};

enum{ /*COMBO_PLAYER combobox values*/
	COMPUTER_BLACK,
	COMPUTER_WHITE,
	COMPUTER_NONE
};

enum{
	TOOLBAR_NEW,
	TOOLBAR_LOAD,
	TOOLBAR_SAVE,
	TOOLBAR_UNDOALL,
	TOOLBAR_UNDO,
	TOOLBAR_REDO,
	TOOLBAR_REDOALL,
	TOOLBAR_HOME,

	TOOLBAR_SIZE
};

enum{
	CHECKBOX_MINIMIZE,
	CHECKBOX_ANIMATION,

	CHECKBOX_SIZE
};

class ReversiFrame : public Base{
	bool m_signals;
public:
	GtkWidget*m_window;
	GtkWidget* m_toolBar;
	GtkToolItem *m_toolBarButton[TOOLBAR_SIZE];
	GtkWidget* m_combo[COMBO_SIZE];
	GtkWidget* m_check[CHECKBOX_SIZE];
	GtkWidget* m_turnList;

	ReversiArea m_area;
	ScoreTurnArea m_scoreTurnArea;
	ReversiFrame();
	void click(GtkWidget *widget);
	void loadConfig();
	void saveConfig();

	inline void setTurnList(std::string s){
		GtkTextBuffer *buffer = gtk_text_view_get_buffer (GTK_TEXT_VIEW (m_turnList));
		gtk_text_buffer_set_text (buffer, s.c_str(), -1);
	}

	inline int getTableSize()const{
		return getComboPosition(COMBO_TABLE_SIZE)+MINIMAL_TABLE_SIZE;
	}

	inline int getShowType()const{
		return getComboPosition(COMBO_SHOW_OPTION);
	}

	inline int getComputerPlayer()const{
		return getComboPosition(COMBO_PLAYER);
	}

	inline int getStartPosition()const{
		return getComboPosition(COMBO_START_POSITION);
	}

	inline int getComboPosition(int i)const{
		return gtk_combo_box_get_active(GTK_COMBO_BOX(m_combo[i]) );
	}

	inline bool isAnimation()const{
		return getCheck(CHECKBOX_ANIMATION);
	}

	inline bool isMinimize()const{
		return getCheck(CHECKBOX_MINIMIZE);
	}

	inline bool getCheck(int i)const{
		return gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON (m_check[i]));
	}

	inline void updateButton(int index,bool enable)const{
		gtk_widget_set_sensitive (GTK_WIDGET(m_toolBarButton[index]),enable);
		gtk_tool_button_set_icon_widget(GTK_TOOL_BUTTON(m_toolBarButton[index]),getToolBarImage(index,enable) );
  	gtk_widget_show_all(m_toolBar);//do not remove!
	}

	inline void setParameters(const int *p){
		m_signals=false;
		gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[COMBO_TABLE_SIZE]),p[0]-MINIMAL_TABLE_SIZE );
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON (m_check[CHECKBOX_MINIMIZE]),p[1]!=0);
		gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[COMBO_START_POSITION]),p[2] );
		m_signals=true;
	}

	GtkWidget* getToolBarImage(int index,bool enable)const;
	static GtkTreeModel* createTextModel(std::string text[]);
	GtkWidget* createTextCombobox(std::string text[]);
	GtkWidget* createPictureCombobox();
	GtkTreeModel* createStartPositionModel();
	void enableWidgets(bool enable);

	void openUrl(const char url[]);

};

#endif /* REVERSIFRAME_H_ */
