import type { MetadataRoute } from "next";
import { allDocs, routeHref } from "@/lib/docs";
import { SITE_URL } from "@/lib/site";

// Written to out/sitemap.xml by the static export.
export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const pages: MetadataRoute.Sitemap = [
    { url: `${SITE_URL}/`, changeFrequency: "weekly", priority: 1 },
    { url: `${SITE_URL}/compare/`, changeFrequency: "monthly", priority: 0.8 },
  ];
  for (const doc of allDocs()) {
    pages.push({
      url: `${SITE_URL}${routeHref(doc.route)}`,
      changeFrequency: "weekly",
      priority: doc.inNav ? 0.7 : 0.4,
    });
  }
  return pages;
}
