// Copy ../docs/assets into public/docs-assets so docs images and the landing page diagram
// are served by the site itself (the repository may not be publicly readable).
import fs from "node:fs";
import path from "node:path";

const src = path.resolve(process.cwd(), "..", "docs", "assets");
const dest = path.resolve(process.cwd(), "public", "docs-assets");
fs.rmSync(dest, { recursive: true, force: true });
fs.cpSync(src, dest, { recursive: true });
console.log(`copied ${path.relative(process.cwd(), src)} -> public/docs-assets`);
