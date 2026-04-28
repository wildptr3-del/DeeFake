# Use official Node.js image
FROM node:18-slim

# Set environment variables
ENV NODE_ENV=production
ENV PORT=5000

# Set work directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install --production

# Copy project files
COPY . .

# Expose the port
EXPOSE 5000

# Start the application
CMD ["node", "server.js"]
