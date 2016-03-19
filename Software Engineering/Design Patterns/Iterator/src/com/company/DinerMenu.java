package com.company;

/**
 * Created by kangth on 2/22/2016.
 */
public class DinerMenu {
    static final int MAM_ITEMS = 6;
    int numberOfItems = 0;
    MenuItem[] menuItems;

    public DinerMenu(){
        menuItems = new MenuItem[MAM_ITEMS];

        addItem("Vegetarian BLT", "Bacon with lettuce & tomato on whole wheat", true, 2.99);
        addItem("BLT", "Bacon with lettuce & tomato on whole wheat", false, 2.99);
        addItem("Soup of the day", "Doup of the day, with a side of potato salad", false, 3.29);
        addItem("Hotdog", "A hot dog, with saurkraut, relish , onions, topped with cheese", false, 3.05);
    }

    public void addItem(String name, String description, boolean vegetarian, double price){

        MenuItem menuItem  = new  MenuItem(name, description, vegetarian, price);

        if(numberOfItems >= MAM_ITEMS){
            System.err.println("sorry, menu is full! can't add item to menu");
        }else{
            menuItems[numberOfItems] = menuItem;
            numberOfItems = numberOfItems + 1;
        }
    }


    public MenuItem[] getMenuItems(){
        return menuItems;
    }

    /*
    public Iterator createIterator(){

        return new DinerMenuIterator(menuItems);
    }
    */
}
