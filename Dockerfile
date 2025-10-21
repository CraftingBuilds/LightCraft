FROM python:3.11-slim

# Working directory
WORKDIR /app

# Copy package files
COPY . /app

# Install LightCraft
RUN pip install .

# Default command
ENTRYPOINT ["lightcraft"]