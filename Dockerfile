FROM python:3.10-slim

LABEL maintainer="luandnh98@gmail.com"

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set workdir and copy app files
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./

# Install dependencies
RUN pip install --upgrade pip

RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Set permissions and switch to non-root
RUN chown -R appuser:appgroup /app
USER appuser

ARG APP_VERSION=unknown
ENV APP_VERSION=$APP_VERSION

# embed version in a file for access in Python
RUN echo $APP_VERSION > /app/VERSION

EXPOSE 5000
CMD ["python", "app.py"]
