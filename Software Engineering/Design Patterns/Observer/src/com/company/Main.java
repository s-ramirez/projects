package com.company;

public class Main {

    public static void main(String[] args) {
	// write your code here

        PersonObserver John = new ConcretePerson("john", "john@wfu.edu");
        PersonObserver Tom = new ConcretePerson("Tom", "tom@wfu.edu");
        PersonObserver James = new ConcretePerson("James", "james@wfu.edu");

        Subject Cs = new ConcreteSubject();

        Cs.registerObserver(John);
        Cs.registerObserver(Tom);
        Cs.registerObserver(James);

        Cs.notifyObserver();

    }
}
