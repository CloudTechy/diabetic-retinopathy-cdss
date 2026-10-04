FROM node:20-alpine

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Start Vite with host 0.0.0.0 so Docker can route traffic
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
