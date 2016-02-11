package com.company;

public class Main {

    public static void main(String[] args) {
	// write your code here
        Shape circle = new Circle();
        System.out.println("normal Circle");
        circle.draw();

        circle = new ShapeColorDecorator(circle, "red");
        System.out.println("\nColored Circle");
        circle.draw();

        circle = new LineColorDecorator(circle , "blue");
        System.out.println("\nColored Circle with line color");
        circle.draw();

        Shape rectangle = new Rectangle();
        rectangle = new ShapeColorDecorator(rectangle, "blue");
        rectangle = new LineColorDecorator(rectangle, "red");
        rectangle.draw();
    }
}
