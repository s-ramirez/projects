package com.company;

/**
 * Created by sramirez on 2/8/16.
 */
public class SortingApp {
    ISorting sortAlg;
    int[] num = {5,2,3,1,6,9,8};

    public void display () {
        for(int i = 0; i < num.length; i++) {
            System.out.print(num[i]);
        }
    }

    public void performSort() {
        sortAlg.sort(num);
    }

    public void setSortAlg(ISorting sort) {
        sortAlg = sort;
    }
}
