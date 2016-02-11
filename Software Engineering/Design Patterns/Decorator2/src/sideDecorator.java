/**
 * Created by kangth on 2/9/2016.
 */

public class sideDecorator extends Display_Decorator {


    char sChar;  // #
    Display display;

    public sideDecorator(Display ds, char ch){
        sChar = ch;
        display = ds;
    }

    public int getColumns(){
        return display.getColumns() + 4;
    }

    public int getRow(){
        return display.getRow() + 0;
    }

    public String getRowText(int row){

        String str;
        str = display.getRowText(row);
        str = " " + str + " ";
        str = sChar + str + sChar;
        return str;

    }


}
