package com.company;

import java.util.ArrayList;

/**
 * Created by kangth on 2/22/2016.
 */
public class PancakeHouseMenu {
    ArrayList menuItems;

    public PancakeHouseMenu(){
        menuItems = new ArrayList();

        addItem("K&B Pancake breakfast", "pancakes with scrambled eggs and toast", true, 2.99);
        addItem("regular pancake breakfast", "pancakes with fried eggs, sausage", false, 2.99);
        addItem("Blueberry pancakes", "pancakes made with fresh blueberries", true, 3.49);
        addItem("Waffles", "waffles, with your choice of blueberries or strawberries", true, 3.59);
    }

    public void addItem(String name, String description, boolean vegetarian, double price){
        MenuItem menuItem = new MenuItem(name, 	description, vegetarian , price);
        menuItems.add(menuItem);
    }

    public ArrayList getMenuItems(){
        return menuItems;
    }

    /*
    public Iterator createIterator(){
        return new PancakeHouseMenuIterator(menuItems);
    }
    */
}
