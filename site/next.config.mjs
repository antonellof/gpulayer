/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: `npm run build` writes a self-contained site to ./out
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
