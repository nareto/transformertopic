import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from random import uniform
from loguru import logger

def scrambleDateColumn(df, dateColumn):
    """
    Returns DataFrame with dates uniformly distributed over a certain period.

    If the original date is the 1st of a certain month, the new date will be randomly picked from that month.

    If it is the 1st of a certain year, the new date will be a random date in that year.
    """

    # changes = 0
    def scramble(row):
        date = row[dateColumn]
        if date.day == 1 and date.month == 1:
            timeDelta = timedelta(days=365)
        elif date.day == 1:
            timeDelta = timedelta(days=30)
        else:
            return row
        endDate = date + timeDelta
        startTimestamp = date.timestamp()
        endTimestamp = endDate.timestamp()
        newTimestamp = uniform(startTimestamp, endTimestamp)
        newDate = dt.fromtimestamp(newTimestamp)
        row[dateColumn] = newDate
        # nonlocal changes
        # changes += 1
        # logger.debug(f"Chnaged date for comment {row['commentid']}: old date: {date}, newdate: {newDate}")
        return row
    # logger.debug(f"Scrambling {dateColumn} for df with columns: {df.columns}")
    retdf = df.apply(scramble, axis=1)
    # logger.debug(f"Made {changes} date changes. Returning df with columns: {retdf.columns}")
    return retdf