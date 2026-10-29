/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for `npm run build` (writes a self-contained site to ./out). Dev mode runs
  // as a normal server, because Next 14's dev server rejects optional catch-all routes
  // under `output: "export"`.
  output: process.env.NODE_ENV === "production" ? "export" : undefined,
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
