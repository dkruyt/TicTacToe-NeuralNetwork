# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install the required packages
RUN pip install --no-cache-dir numpy tensorflow matplotlib colorama ipython

# Command to run the script, allowing arguments to be passed
ENTRYPOINT ["python", "tictactoe.py"]
