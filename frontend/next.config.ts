import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    middlewareClientMaxBodySize: 524288000, // 500 MB — for middleware
  },
  // Disable Next.js body parsing for API proxy routes so large uploads
  // stream directly to the backend without buffering
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
