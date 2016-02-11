package com.company;

/**
 * Created by kangth on 2/9/2016.
 */
public class ShapeColorDecorator extends ShapeDecorator{

    private String ShapeColor;

    public ShapeColorDecorator(Shape decoratedShape, String SColor) {
        super(decoratedShape);
        this.ShapeColor = SColor;
    }

    @Override
    public void draw() {
        decoratedShape.draw();
        System.out.println("Shape Color: " + this.ShapeColor);
    }


}
