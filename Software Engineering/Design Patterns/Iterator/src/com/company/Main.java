package com.company;

public class Main {

    public static void main(String[] args) {
	// write your code here
        // write your code here
        PancakeHouseMenu pancakeHouseMenu = new PancakeHouseMenu();
        DinerMenu dinerMenu = new DinerMenu();

        Waitress waitress = new Waitress(pancakeHouseMenu, dinerMenu);
        waitress.printMenu();

    }
}
