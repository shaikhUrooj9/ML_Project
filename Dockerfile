# Step 1: Switch to the full standard Python image (fixes package installation errors)
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy requirements file first to build dependencies
COPY requirements.txt .

# Step 4: Install Python packages safely
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project fissles into the container
COPY . .

# Step 6: Expose port 5000
EXPOSE 5000

# Step 7: Run the application
CMD ["python", "app.py"]