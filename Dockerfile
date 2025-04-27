FROM python:3.10-slim

LABEL maintainer="luandnh98@gmail.com"

# Set workdir and copy app files
WORKDIR /app

# Install build essentials for sklearn, xgboost, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt ./
COPY app.py ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Declare ARG and ENV
ARG APP_VERSION=unknown
ENV APP_VERSION=${APP_VERSION}

# Write version to file during build (use ARG instead of ENV)
RUN echo "${APP_VERSION}" > /app/VERSION

EXPOSE 8000

CMD ["python", "app.py"]