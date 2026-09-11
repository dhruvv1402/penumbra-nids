import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // The API runs separately (uvicorn on 8000). Proxying through Next keeps the browser on one
  // origin, which avoids CORS entirely in development and means the demo works with the API
  // reachable only on localhost.
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};

export default config;
