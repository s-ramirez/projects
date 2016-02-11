/**
 * Created by kangth on 2/9/2016.
 */

public abstract class Display {


    public void DrawScreen(){

        System.out.println(getRow());

        for(int i=0; i< getRow(); i++){
            System.out.println(getRowText(i));
        }

    }
    public abstract int getColumns();
    public abstract int getRow();
    public abstract String getRowText(int row);

}
