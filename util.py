import json

def get_train_time(name):
    """
    Function for getting and formatting the train time of a model
    Params:
        name: name of the model
    """

    # Get the train time from the specified input file
    path = f'./stats/{name}/stats.json'

    with open(path, "r") as file:
        data = json.load(file)
    
    train_time = data["train_time"]

    # Calculate number of minutes and seconds
    train_time_minutes = int(train_time // 60)
    train_time_seconds = int(train_time - train_time_minutes * 60)

    # Format and print total time
    total_time = f"{train_time_minutes} minutes, {train_time_seconds} seconds"

    print(f"Total train time for model {name} is {total_time}.")

    return (train_time_minutes, train_time_seconds)