FROM node:22-alpine

# Install ffmpeg
RUN apk add --no-cache ffmpeg

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY . .

RUN mkdir -p /data/uploads /data/outputs

ENV UPLOAD_DIR=/data/uploads
ENV OUTPUT_DIR=/data/outputs
ENV PORT=8000

EXPOSE 8000

CMD ["node", "server.js"]
