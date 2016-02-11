package com.company;

public class Main {

    public static void main(String[] args) {
	// write your code here
        SortingApp bubble = new BubbleApp();
        bubble.setSortingAlg(new SelectionSort());
        bubble.performSort();
        bubble.display();


    }
}
