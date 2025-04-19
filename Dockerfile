FROM python:3.10-slim

LABEL maintainer="luandnh98@gmail.com"

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set working directory
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./
COPY model_comparison_results.json ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Ensure permissions for non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]