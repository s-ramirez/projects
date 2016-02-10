package com.company;

/**
 * Created by sramirez on 2/8/16.
 */
public class SelectionSort implements ISorting {
    public void sort(int[] num) {
        System.out.println("selectinSort");
        for (int i = 0; i < num.length - 1; i++)
        {
            int index = i;
            for (int j = i + 1; j < num.length; j++)
                if (num[j] < num[index])
                    index = j;
            int smallerNumber = num[index];
            num[index] = num[i];
            num[i] = smallerNumber;
        }
    }
}
