# Penumbra console image.
#
# Next.js standalone output, so the runtime layer carries the server and the traced dependencies
# rather than the whole of node_modules. The difference is roughly an order of magnitude, and on a
# machine with 21 GB free that matters more than elegance.

FROM node:24-alpine AS deps

WORKDIR /app

# Dependencies in their own layer: they change far less often than the pages, so editing a
# component does not reinstall the tree.
COPY package.json package-lock.json* ./
RUN npm ci


FROM node:24-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Telemetry is off during the build as well as at runtime. A build step that phones home is a build
# step that fails behind a conference firewall.
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build


FROM node:24-alpine AS runtime

# Runs unprivileged. A static frontend has no reason to be root, and node:alpine already ships a
# `node` user, so there is nothing to create.
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# `standalone` traces exactly the files the server needs. `.next/static` is NOT part of that trace
# and has to be copied alongside it - omitting it is the usual way this image ends up serving a
# page with no CSS. There is no public/ directory in this project; add the copy if one appears.
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static

USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "require('http').get('http://127.0.0.1:3000/', r => process.exit(r.statusCode < 500 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
