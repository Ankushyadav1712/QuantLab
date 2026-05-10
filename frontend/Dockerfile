# ---------- stage 1: build the Vite bundle ----------
# Vite 8 (already pinned in package.json) requires Node >=20.19, so we use
# node:20-alpine instead of the 18-alpine the original spec called for.
FROM node:26-alpine AS builder

WORKDIR /app

# Vite reads VITE_* env vars at build time and inlines them into the bundle.
# Pass through `--build-arg VITE_API_URL=...` (docker-compose handles this).
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------- stage 2: nginx serves the static bundle ----------
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
