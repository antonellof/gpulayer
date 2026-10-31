// Canonical origin of the public site; metadata, the sitemap and robots.txt build URLs from it.
export const SITE_URL = "https://www.gpulayer.dev";

export const SITE_NAME = "GPUlayer";

export const SITE_DESCRIPTION =
  "Train PyTorch on every GPU you can reach: one Trainer, from your laptop to many machines. NVIDIA CUDA, Apple Metal and CPU; open source and self-hosted.";

// The image link previews show (1200 x 630). public/og.png, made with next/og's ImageResponse.
export const OG_IMAGE = {
  url: "/og.png",
  width: 1200,
  height: 630,
  alt: "GPUlayer: train PyTorch on every GPU you can reach",
};

// Google Analytics 4 measurement id for www.gpulayer.dev.
export const GA_ID = "G-WE1BJY7J3N";
