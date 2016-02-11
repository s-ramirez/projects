/**
 * Created by kangth on 2/9/2016.
 */
public class Main {

    public static void main(String[] args){

        Display ds = new TextDisplay("Hello World");

        ds = new sideDecorator(ds, '#');

        //ds = new sideDecorator(ds, '+');
        //ds = new fullDecorator(ds, '-');
        //ds = new sideDecorator(ds, '*');
        //ds = new fullDecorator(ds, '+');
        //ds = new fullDecorator(ds, '#');

        ds.DrawScreen();

    }
}
