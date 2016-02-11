package com.company;

/**
 * Created by kangth on 2/9/2016.
 */
public abstract class ShapeDecorator implements Shape{

    protected Shape decoratedShape;

    public ShapeDecorator(Shape decoratedShape){
        this.decoratedShape = decoratedShape;
    }

    public abstract void draw();

}
