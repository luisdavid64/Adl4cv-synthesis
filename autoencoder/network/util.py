
def filter_data(data, filter_label="desk"):
    return filter(lambda x: x["label"] == filter_label,data)