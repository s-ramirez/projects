package com.company;

import java.util.ArrayList;

/**
 * Created by kangth on 2/22/2016.
 */
public class PancakeHouseMenuIterator {
    ArrayList menuItems;
    int position = 0;

    public PancakeHouseMenuIterator(ArrayList menuItems){
        this.menuItems = menuItems;
    }

    public Object next(){
        Object  menuItem =  menuItems.get(position);
        position =position +1;
        return menuItem;
    }

    public boolean hasNext(){
        if(position >= menuItems.size() || menuItems.get(position) == null){
            return false;
        }else{
            return true;
        }
    }
}
