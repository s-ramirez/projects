package com.company;

/**
 * Created by kangth on 3/2/2016.
 */
public class PasswordValidator {

    public boolean isValid(String password){

            if(password.length() >= 6 && password.length() <= 10){
                return true;
            }else{
                return false;
            }
    }


    public int max(int a, int b){

        if(a > b){
            return a;
        }else{
            return b;
        }

    }

}
