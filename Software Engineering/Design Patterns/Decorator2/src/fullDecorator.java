/**
 * Created by kangth on 2/9/2016.
 */

public class fullDecorator extends Display_Decorator{

    char sChar;
    Display display;

    public fullDecorator(Display ds, char ch){
        sChar = ch;
        display = ds;
    }


    public int getColumns(){
        return display.getColumns() + 2;
    }

    public int getRow(){

        return display.getRow() + 2;
    }

    public String getRowText(int row){

        if(row == 0){
            return sChar + makeline(display.getColumns()) + sChar;

        }else if(row == display.getRow() +1){
            return sChar + makeline(display.getColumns()) + sChar;

        }else{
            return sChar + display.getRowText(row - 1)+ sChar;
        }


    }

    public String makeline(int count){

        String str="";
        for(int i=0; i < count ; i++){
            str = str + sChar;
        }
        return str;
    }

}
