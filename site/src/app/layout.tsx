import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import { THEME_SCRIPT } from "@/components/site/theme-toggle";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });
const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", display: "swap" });

export const metadata: Metadata = {
  title: {
    default: "GpuLayer | Run any GPU workload on any GPU you can reach",
    template: "%s | GpuLayer",
  },
  description:
    "A workload-agnostic GPU layer. Run commands in any language, Python functions with any library, and distributed jobs on your laptop, workstations, servers or cloud VMs, with a deeper integration for PyTorch. Pre-alpha.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable} ${display.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen bg-background font-sans">{children}</body>
    </html>
  );
}
