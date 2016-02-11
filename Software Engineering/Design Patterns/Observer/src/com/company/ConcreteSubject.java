package com.company;

import java.util.Vector;
import java.util.Enumeration;

public class ConcreteSubject implements Subject {

    private Vector listeners = new Vector();

    public void registerObserver(PersonObserver po){
        listeners.addElement(po);
    }
    public void removeObserver(PersonObserver po){
        listeners.remove(po);
    }
    public void notifyObserver(){
        for (Enumeration e = listeners.elements(); e.hasMoreElements();)
            ((PersonObserver) e.nextElement()).update();
    }


}
