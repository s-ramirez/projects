package com.company;

/**
 * Created by kangth on 2/7/2016.
 */

public class SortingApp {
    ISorting sortAlg;
    int[] num = {5,4,3,2,1,6,8,9};

    public void performSort(){
        sortAlg.sort(num);
    }

    public void setSortingAlg(ISorting sort){
        sortAlg = sort;
    }
    public void display(){

        for (int i = 0; i < num.length - 1; i++) {
            System.out.print(num[i]);

        }
        System.out.println("\n");
    }

}
