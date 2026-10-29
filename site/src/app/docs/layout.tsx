import type { Metadata } from "next";
import { DocsSidebar, type SidebarSection } from "@/components/docs/sidebar";
import { SiteHeader } from "@/components/site/header";
import { SiteFooter } from "@/components/site/footer";
import { readNav, routeHref, slugToRoute } from "@/lib/docs";

export const metadata: Metadata = {
  title: { default: "Docs", template: "%s | GPUlayer docs" },
};

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  const sections: SidebarSection[] = readNav().map((s) => ({
    title: s.title,
    pages: s.pages.map((p) => ({ title: p.title, href: routeHref(slugToRoute(p.slug)) })),
  }));
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <div className="mx-auto w-full max-w-[90rem] flex-1 lg:flex">
        <DocsSidebar sections={sections} />
        <div className="min-w-0 flex-1">{children}</div>
      </div>
      <SiteFooter />
    </div>
  );
}
