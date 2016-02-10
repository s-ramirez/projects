package com.company;

public class Main {

    public static void main(String[] args) {
        SortingApp bubble = new BubbleApp();
        bubble.performSort();
        bubble.display();
        System.out.println();

        bubble.setSortAlg(new SelectionSort());
        bubble.performSort();
        bubble.display();
    }
}
