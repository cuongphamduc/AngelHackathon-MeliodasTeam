package com.vit.trafficvolumemap.ui.util;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class DateTimePickerHelper {
    public static final String DATE_PICKER_FRAGMENT_DIALOG = "DatePickerFragmentDialog";
    public static final String TIME_PICKER_FRAGMENT_DIALOG = "TimePickerFragmentDialog";

    /**
     * all is format time
     *
     * @param time
     * @return string time formated
     */

    public static String getDateOnly(long time) {
        return new SimpleDateFormat("dd MMM yyyy", Locale.getDefault()).format(time);
    }

    public static String getDateAndTime(long time) {
        SimpleDateFormat sample = new SimpleDateFormat("dd/MM/yyyy, HH:mm", Locale.getDefault());
        return sample.format(new Date(time));
    }

    public static String getDateTime(long time) {
        SimpleDateFormat sample = new SimpleDateFormat("yyyyMMddHHmm", Locale.getDefault());
        return sample.format(new Date(time));
    }

    public static String getTimeOnly(long time) {
        SimpleDateFormat sample = new SimpleDateFormat("hh:mma", Locale.getDefault());
        return sample.format(time);
    }
}
