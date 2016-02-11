package com.company;

/**
 * Created by kangth on 2/9/2016.
 */
public class LineColorDecorator extends ShapeDecorator{
    private String lineColor;

    public LineColorDecorator(Shape decoratedShape , String lColor) {
        super(decoratedShape);
        this.lineColor = lColor;
    }

    @Override
    public void draw() {
        decoratedShape.draw();
        System.out.println("line Color: " + this.lineColor);
    }


}
