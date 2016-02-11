package com.company;

/**
 * Created by kangth on 2/7/2016.
 */
public interface Subject {

    public void registerObserver(PersonObserver po);
    public void removeObserver(PersonObserver po);
    public void notifyObserver();
}
