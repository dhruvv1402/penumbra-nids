import type { NextConfig } from "next";

// Where the API lives. Localhost for `npm run dev`; the compose service name inside a container,
// where 127.0.0.1 is the console's own loopback and nothing answers on it.
const API = process.env.PENUMBRA_API_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  reactStrictMode: true,
  // Traced output: the runtime image carries the server plus the files it actually imports, rather
  // than the whole of node_modules.
  output: "standalone",
  // The API runs separately (uvicorn on 8000). Proxying through Next keeps the browser on one
  // origin, which avoids CORS entirely in development and means the demo works with the API
  // reachable only on localhost.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};

export default config;
