package com.company;

import java.util.ArrayList;

/**
 * Created by kangth on 2/22/2016.
 */
public class Waitress {
    PancakeHouseMenu pancakeHouseMenu = new PancakeHouseMenu();
    DinerMenu dinerMenu = new DinerMenu();

    public Waitress(PancakeHouseMenu pancakeHouseMenu, DinerMenu dinerMenu) {
        this.pancakeHouseMenu = pancakeHouseMenu;
        this.dinerMenu = dinerMenu;
    }

    public void printMenu() {

        ArrayList breakfastItems = pancakeHouseMenu.getMenuItems();
        for(int i = 0; i < breakfastItems.size(); i++) {
            MenuItem menuItem = (MenuItem)breakfastItems.get(i);
            System.out.print(menuItem.getName());
            System.out.println("\t\t" + menuItem.getPrice());
            System.out.println("\t" + menuItem.getDescription());
        }

        MenuItem[] lunchItems = dinerMenu.getMenuItems();

        for(int i = 0; i < lunchItems.length ; i++) {
            MenuItem menuItem = lunchItems[i];
            System.out.print(menuItem.getName());
            System.out.println("\t\t" + menuItem.getPrice());
            System.out.println("\t" + menuItem.getDescription());
        }


    }

    /*
    public void printMenu() {
        Iterator pancakeIterator = pancakeHouseMenu.createIterator();
        Iterator dinerIterator = dinerMenu.createIterator();
        System.out.println("Menu\n----\nBREAKFAST");
        printMenu(pancakeIterator);
        System.out.println("Menu\n------\nDINNER");
        printMenu(dinerIterator);
    }

    private void printMenu(Iterator iterator) {
        while (iterator.hasNext()) {
            MenuItem menuItem = (MenuItem)iterator.next();
            System.out.print(menuItem.getName() + ", ");
            System.out.print(menuItem.getPrice() + " -- ");
            System.out.println(menuItem.getDescription());
        }
    }

    */
}
