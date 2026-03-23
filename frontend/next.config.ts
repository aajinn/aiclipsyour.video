import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    middlewareClientMaxBodySize: 524288000, // 500 MB
  },
};

export default nextConfig;
