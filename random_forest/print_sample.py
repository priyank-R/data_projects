import seaborn as sns

def print_sample():
    # Load data
    data = sns.load_dataset("mpg") 
    print(data.sample(10))

print_sample()