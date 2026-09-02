# Start from a clean Python 3.11 environment
FROM python:3.11-slim

# Set the working folder inside the box
WORKDIR /app

# Copy the list of needed libraries first (helps Docker cache this step)
COPY requirements.txt .

# Install all the libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our project files into the box
COPY . .

# Tell Docker which port Streamlit uses
EXPOSE 8501

# The command that runs when the box starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]