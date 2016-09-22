
def count_list_and_int(input_series):
""" This functions counts the elements in each element contained in a series of
    lists. If the element is an integer then the count is set to 0"""

    res = [0]*len(input_series)
    for i, lst in enumerate(input_series): #data_active
        if isinstance(lst, (int, long)):
            res[i] = 0
        else:
            res[i] = len(lst)

    return res


def deduplicate_list(input_series):
""" This functions counts takes a series of lists and remove the duplicated
    values. If the element is an integer then the count is set to 0"""

    res = [0]*len(input_series)
    for i, lst in enumerate(input_series): #data_active
        if isinstance(lst, (int, long)):
            res[i] = 0
        else:
            res[i] = list(set(lst))

    return res


def most_common_in_list(input_series, placement):
""" This functions takes a series of lists and returns the element most frequent
    in each list. If the element is an integer then the count is set to 0"""

    from collections import Counter
    
    res = [0]*len(input_series)
    for i, lst in enumerate(input_series): #data_active
        if isinstance(lst, (int, long)):
            res[i] = 0
        else:
            count = Counter(lst)
            if len(count) < placement:
                res[i] = 'nan'
            else:
                res[i] = count.most_common()[placement-1][0]
    return res
