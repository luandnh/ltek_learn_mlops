FROM python:3.10-slim

LABEL maintainer="luandnh98@gmail.com"

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set workdir and copy app files
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Set permissions
RUN chown -R appuser:appgroup /app

# Declare ARG and ENV
ARG APP_VERSION=unknown
ENV APP_VERSION=${APP_VERSION}

# Write version to file during build (use ARG instead of ENV)
RUN echo "${APP_VERSION}" > /app/VERSION

# Switch to non-root user
USER appuser

EXPOSE 5000
CMD ["python", "app.py"]
