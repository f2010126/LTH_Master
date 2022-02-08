import os

# Check if the directory exist and if not, create a new directory
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)