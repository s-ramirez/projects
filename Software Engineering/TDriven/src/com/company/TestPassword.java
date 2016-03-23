package com.company;

/**
 * Created by kangth on 3/2/2016.
 */
public class TestPassword {

    public void testValidator(){
        PasswordValidator pv = new PasswordValidator();
        assertEquals(false, pv.isValid("Ab123"));

        assertEquals(false, pv.isValid("abc12"));
        assertEquals(false, pv.isValid("abcdef"));

        assertEquals(6, pv.max(6,2));
        assertEquals(6, pv.max(2,6));
    }
}
