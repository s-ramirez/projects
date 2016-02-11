/**
 * Created by kangth on 2/9/2016.
 */

public class TextDisplay extends Display{


    private String str;

    public TextDisplay(String str){
        this.str = str;
    }

    public int getColumns(){
        return str.length();
    }

    public int getRow(){
        return 1;
    }

    public String getRowText(int row){

        if(row == 0){
            return str;
        }else{

            return "";
        }

    }

}

