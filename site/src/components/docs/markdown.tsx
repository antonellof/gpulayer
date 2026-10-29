import { isValidElement, type ReactElement, type ReactNode } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "@/components/docs/code-block";
import { Mermaid } from "@/components/docs/mermaid";
import { resolveHref } from "@/lib/docs";

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement(node)) return textOf((node.props as { children?: ReactNode }).children);
  return "";
}

/** Server-rendered Markdown for one docs file; ``file`` is relative to docs/. */
export function Markdown({ content, file }: { content: string; file: string }) {
  return (
    <div className="docs-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSlug]}
        components={{
          a: ({ href = "", children }) => {
            const to = resolveHref(href, file);
            if (/^https?:/.test(to)) {
              return (
                <a href={to} target="_blank" rel="noreferrer">
                  {children}
                </a>
              );
            }
            if (to.startsWith("#") || to.startsWith("mailto:")) return <a href={to}>{children}</a>;
            return <Link href={to}>{children}</Link>;
          },
          pre: ({ children }) => {
            const child = (Array.isArray(children) ? children[0] : children) as ReactElement<{
              className?: string;
              children?: ReactNode;
            }>;
            const className = isValidElement(child) ? child.props.className ?? "" : "";
            const lang = /language-([\w-]+)/.exec(className)?.[1];
            const code = textOf(isValidElement(child) ? child.props.children : children);
            if (lang === "mermaid") return <Mermaid chart={code.trim()} />;
            return <CodeBlock code={code} language={lang} />;
          },
          img: ({ src, alt }) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={typeof src === "string" ? resolveHref(src, file).replace("/blob/", "/raw/") : undefined}
              alt={alt ?? ""}
              className="rounded-lg border border-border"
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
