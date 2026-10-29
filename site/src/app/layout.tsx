import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme-toggle";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: {
    default: "GpuLayer | Run PyTorch on any GPU you can reach",
    template: "%s | GpuLayer",
  },
  description:
    "Open-source GPU execution and orchestration for PyTorch. One training file runs on CPU, Apple MPS or NVIDIA CUDA, locally, on remote machines or across nodes. Pre-alpha.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen bg-background font-sans">{children}</body>
    </html>
  );
}
