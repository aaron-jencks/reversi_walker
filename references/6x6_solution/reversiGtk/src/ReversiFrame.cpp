/*
 * ReversiFrame.cpp
 *
 *  Created on: 25.01.2015
 *      Author: alexey slovesnov
 */

#include "ReversiFrame.h"

void button_clicked(GtkWidget *widget, ReversiFrame*panel){
	panel->click(widget);
}

static void combo_changed(GtkComboBox *comboBox, ReversiFrame*frame) {
	frame->click( (GtkWidget*)comboBox );
}

static void toggle_check(GtkWidget *widget, ReversiFrame*frame){
	frame->click(widget);
}

static void  destroy_event(ReversiFrame *frame, gpointer user_data){
	frame->saveConfig();
	gtk_main_quit();
}

static std::string SHOW_STRING[]={"show possible moves","show estimates","show nothing",""};
static std::string PLAYER_STRING []={"computer is black","computer is white","two human players",""};

ReversiFrame::ReversiFrame() {
	int i;
	char c[16];
  GtkWidget *w,*w1;
  BaseArea::m_reversiFrame=this;
  m_signals=true;

  m_toolBar=gtk_toolbar_new();
	gtk_toolbar_set_style(GTK_TOOLBAR(m_toolBar), GTK_TOOLBAR_ICONS);
	gtk_container_set_border_width(GTK_CONTAINER(m_toolBar), 0);

	for(i=0;i<SIZE(m_toolBarButton);i++){
		m_toolBarButton[i] = gtk_tool_button_new(getToolBarImage(i,true),"");
		gtk_toolbar_insert(GTK_TOOLBAR(m_toolBar), m_toolBarButton[i], -1);
		g_signal_connect(m_toolBarButton[i], "clicked",G_CALLBACK(button_clicked), this );
	}

  w = gtk_box_new (GTK_ORIENTATION_VERTICAL, 2);

  m_combo[COMBO_SHOW_OPTION]=createTextCombobox(SHOW_STRING);
  gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[COMBO_SHOW_OPTION]),1);
  gtk_box_pack_start (GTK_BOX(w),m_combo[COMBO_SHOW_OPTION], FALSE, FALSE, 0);

	m_combo[COMBO_PLAYER]=createTextCombobox(PLAYER_STRING);
  gtk_box_pack_start (GTK_BOX(w),m_combo[COMBO_PLAYER], FALSE, FALSE, 0);

  //table size
  std::string TABLE_SIZE_STRING[5];
  for(i=0;i<4;i++){
  	sprintf(c,"%dx%d",i+MINIMAL_TABLE_SIZE,i+MINIMAL_TABLE_SIZE);
  	TABLE_SIZE_STRING[i]=c;
  }
  TABLE_SIZE_STRING[i]="";
  m_combo[COMBO_TABLE_SIZE]=createTextCombobox(TABLE_SIZE_STRING);
  gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[COMBO_TABLE_SIZE]),2);

  //start position
  w1 = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 3);
	m_combo[COMBO_START_POSITION] =createPictureCombobox();
	gtk_combo_box_set_model(GTK_COMBO_BOX(m_combo[COMBO_START_POSITION]),createStartPositionModel());
  gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[COMBO_START_POSITION]),0);

	//check
  for(i=0;i<CHECKBOX_SIZE;i++){
		m_check[i]=gtk_check_button_new_with_label(i==0?"minimize":"animation");
		if(i==CHECKBOX_ANIMATION){//by default minimize checkbox is active
			gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON (m_check[i]),TRUE);
		}
  }

  //table size, start position, minimize, animation to grid
  w1 = gtk_grid_new ();
  gtk_grid_set_column_spacing(GTK_GRID(w1),3);
  gtk_grid_set_row_spacing(GTK_GRID(w1), 2 );

  for(i=0;i<2;i++){
  	gtk_grid_attach (GTK_GRID (w1),gtk_label_new (i==0?"table":"start"), 0, i,1,1);
  	gtk_grid_attach (GTK_GRID (w1),m_combo[i==0?COMBO_TABLE_SIZE:COMBO_START_POSITION], 1, i,1,1);
  	gtk_grid_attach (GTK_GRID (w1),m_check[i], 2, i,1,1);
  }
  gtk_box_pack_start (GTK_BOX(w),w1, FALSE, FALSE, 0);

  //score & turn
  gtk_box_pack_start (GTK_BOX(w),m_scoreTurnArea.getArea(), FALSE, FALSE, 0);

	m_turnList=gtk_text_view_new ();
	gtk_text_view_set_editable(GTK_TEXT_VIEW(m_turnList),FALSE);
	gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(m_turnList),FALSE);

	w1=gtk_scrolled_window_new(NULL, NULL);

	gtk_container_add(GTK_CONTAINER(w1), m_turnList);
  gtk_box_pack_start (GTK_BOX(w),w1, TRUE, TRUE, 0);
//	gtk_box_pack_start (GTK_BOX(w),m_turnList, TRUE, TRUE, 0);

  w1=gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 3);;
  gtk_box_pack_start (GTK_BOX(w1), m_area.getArea(), FALSE, FALSE, 0);
  gtk_box_pack_start (GTK_BOX(w1), w, FALSE, FALSE, 0);

  w = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
  gtk_box_pack_start (GTK_BOX(w), m_toolBar, FALSE, FALSE, 0);
  gtk_box_pack_start (GTK_BOX(w), w1, FALSE, FALSE, 0);


  loadConfig();
  m_area.newGame();//call after load config! because it set computer player

  //now we can connect signals //Note after set active in loadConfig()
  for(i=0;i<COMBO_SIZE;i++){
  	g_signal_connect(m_combo[i], "changed",G_CALLBACK(combo_changed), gpointer(this) );
  }
  for(i=0;i<CHECKBOX_SIZE;i++){
		g_signal_connect (m_check[i], "toggled",G_CALLBACK (toggle_check), gpointer(this) );
  }

	m_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  gtk_window_set_position(GTK_WINDOW(m_window), GTK_WIN_POS_CENTER);
  gtk_window_set_resizable(GTK_WINDOW(m_window),false);
  gtk_window_set_title(GTK_WINDOW(m_window),"reversi");

  /*---------------- CSS ----------------------------------------------------------------------------------------------------*/
  GtkCssProvider *provider;
  GdkDisplay *display;
  GdkScreen *screen;

	provider = gtk_css_provider_new ();
	display = gdk_display_get_default ();
	screen = gdk_display_get_default_screen (display);
	gtk_style_context_add_provider_for_screen (screen,GTK_STYLE_PROVIDER(provider),GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);

	gsize bytes_written, bytes_read;

	const gchar* home = "reversi.css";
	GError *error = 0;
	gtk_css_provider_load_from_path (provider,g_filename_to_utf8(home, strlen(home), &bytes_read, &bytes_written, &error),NULL);
  g_object_unref (provider);
  /*-------------------------------------------------------------------------------------------------------------------------*/

  g_signal_connect_swapped(G_OBJECT(m_window), "destroy",G_CALLBACK(destroy_event), this );

  gtk_container_add(GTK_CONTAINER(m_window), w);
  gtk_widget_show_all(m_window);

  gtk_main();

}

void ReversiFrame::click(GtkWidget *widget){
	if(!m_signals){
		return;
	}

	GtkToolItem **p=std::find(m_toolBarButton,m_toolBarButton+SIZE(m_toolBarButton), (GtkToolItem *)widget );
	if(p==m_toolBarButton+SIZE(m_toolBarButton)){
		if( widget==m_combo[COMBO_TABLE_SIZE] || widget==m_combo[COMBO_START_POSITION] || widget==m_check[CHECKBOX_MINIMIZE]){
			m_area.newGame();
		}
		else if(widget==m_combo[COMBO_SHOW_OPTION]){
			if(getShowType()==SHOW_TYPE_ESTIMATES){
				m_area.showTypeEstimatesOn();
			}
			else{
				m_area.draw();
			}
		}
	}
	else{
		int index=p-m_toolBarButton;
		if(index==TOOLBAR_NEW){
			m_area.newGame();
		}
		else if(index==TOOLBAR_LOAD){
			m_area.load();
		}
		else if(index==TOOLBAR_SAVE){
			m_area.save();
		}
		else if(index>=TOOLBAR_UNDOALL && index<=TOOLBAR_REDOALL){
			m_area.undoRedo(index);
		}
		else if(index==TOOLBAR_HOME){
			openUrl(HOMEPAGE);
		}
	}
}

GtkTreeModel* ReversiFrame::createTextModel(std::string text[]) {
	GtkTreeIter iter;
	GtkTreeStore *store;
	std::string*p;

	store = gtk_tree_store_new(1, G_TYPE_STRING);
	for( p=text ; *p!="" ; p++) {
		gtk_tree_store_append(store, &iter, NULL);
		gtk_tree_store_set(store, &iter,0, p->c_str() ,-1);
	}
	return GTK_TREE_MODEL(store);

}

GtkWidget* ReversiFrame::createTextCombobox(std::string text[]) {
	GtkWidget*w=gtk_combo_box_new_with_model(createTextModel(text));
  gtk_combo_box_set_active(GTK_COMBO_BOX(w), 0);
  GtkCellRenderer *renderer = gtk_cell_renderer_text_new();
	gtk_cell_layout_pack_start(GTK_CELL_LAYOUT(w), renderer, FALSE);
	gtk_cell_layout_set_attributes(GTK_CELL_LAYOUT(w), renderer,"text", 0,NULL);
	g_object_set (G_OBJECT (renderer),"font", "Times New Roman, 14",NULL);
	return w;
}

GtkWidget* ReversiFrame::createPictureCombobox() {
	GtkWidget*combo= gtk_combo_box_new ();
	GtkCellRenderer *renderer;
	renderer = gtk_cell_renderer_pixbuf_new();
	gtk_cell_layout_pack_start(GTK_CELL_LAYOUT(combo), renderer, FALSE);
	gtk_cell_layout_set_attributes(GTK_CELL_LAYOUT(combo), renderer,"pixbuf", 0,NULL);
	return combo;
}

GtkTreeModel* ReversiFrame::createStartPositionModel() {
	const int size=34;
	GdkPixbuf *pixbuf;
	GtkTreeIter iter;
	GtkTreeStore *store;
	int i,j;
  cairo_t *cr;
  cairo_surface_t *surface;

	store = gtk_tree_store_new(1, GDK_TYPE_PIXBUF);
  for( i=0 ; i <  SIZE(START_POSITION)  ; i++ ){
    const int*p=START_POSITION[i];
  	surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 2*m_cellSize, 2*m_cellSize);
  	cr = cairo_create(surface);
  	for(j=0;j<4;j++){
  		copy(chip(std::find(p,p+2,j)!=p+2),cr,j%2 ? m_cellSize:0 , j/2 ? m_cellSize:0 );
  	}

  	pixbuf=gdk_pixbuf_get_from_surface(surface,0,0,2*m_cellSize,2*m_cellSize);
  	pixbuf=gdk_pixbuf_scale_simple(pixbuf,size,size,GDK_INTERP_BILINEAR);

//  	const char ext[]="png";
//  	char c[16];
//  	sprintf(c,"start%d.%s",i,ext);
//  	gdk_pixbuf_save (pixbuf,c , ext,NULL, NULL);

		gtk_tree_store_append(store, &iter, NULL);
		gtk_tree_store_set(store, &iter,0, pixbuf,-1);
		g_object_unref(pixbuf);
		destroy(cr);
		destroy(surface);
	}

	return GTK_TREE_MODEL(store);
}

GtkWidget* ReversiFrame::getToolBarImage(int index, bool enable) const {
	GdkPixbuf* p=pixbuf("t%d.png", index );
	if(!enable){
		gdk_pixbuf_saturate_and_pixelate (p,p,0.3f,false);//desaturate image
	}
	return gtk_image_new_from_pixbuf(p);
}

void ReversiFrame::loadConfig() {
  FILE*f=fopen(CONFIG_FILENAME,"r");
  if(f){
		int i,v;
		for(i=0;i<COMBO_SIZE;i++){
			fscanf(f,"%d",&v);
			gtk_combo_box_set_active(GTK_COMBO_BOX(m_combo[i]),v );
		}
		for(i=0;i<CHECKBOX_SIZE;i++){
			fscanf(f,"%d",&v);
			gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON (m_check[i]),v);
		}
  	fclose(f);
  }

}

void ReversiFrame::saveConfig() {
	FILE*f=fopen(CONFIG_FILENAME,"w+");
	if(f){
		int i;
		for(i=0;i<COMBO_SIZE;i++){
			fprintf(f,"%d ",getComboPosition(i) );
		}
		for(i=0;i<CHECKBOX_SIZE;i++){
			fprintf(f,"%d ",getCheck(i) );
		}
	}
}

void ReversiFrame::enableWidgets(bool enable) {
	int i;
	for(i=0;i<SIZE(m_toolBarButton);i++){
		updateButton(i,enable);
	}
	for(i=0;i<COMBO_SIZE;i++){
		gtk_widget_set_sensitive (GTK_WIDGET(m_combo[i]),enable);
	}
	for(i=0;i<CHECKBOX_SIZE;i++){
		gtk_widget_set_sensitive (GTK_WIDGET(m_check[i]),enable);
	}
}

void ReversiFrame::openUrl(const char url[]) {
	gtk_show_uri_on_window (0, url,gtk_get_current_event_time (),NULL);
}
