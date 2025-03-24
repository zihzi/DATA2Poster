def month_patch(month):
    for k, v in {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }.items():
        if k in month.lower():
            return v
    raise ValueError(f"Invalid month: {month}")