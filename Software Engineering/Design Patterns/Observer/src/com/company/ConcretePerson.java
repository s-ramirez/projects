package com.company;

/**
 * Created by kangth on 2/7/2016.
 */
public class ConcretePerson implements PersonObserver{

    String personName;
    String email;

    public ConcretePerson(String personName, String email){
        this.personName = personName;
        this.email = email;
    }

    public void update(){

        System.out.println("send an email to " + personName);
    }

}
