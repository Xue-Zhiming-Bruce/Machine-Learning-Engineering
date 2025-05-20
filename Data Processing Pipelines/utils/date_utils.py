from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Generate a list of first-of-month date strings between start_date_str and end_date_str (inclusive).
    Dates are in 'YYYY-MM-DD' format.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return dates