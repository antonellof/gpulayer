"use client";

import { useEffect, useRef } from "react";
import { useIsDark } from "@/lib/use-dark";

/**
 * Wireframe landscape behind the hero: mountains, lakes and hills drawn as a triangle grid,
 * flowing slowly toward the viewer. three.js is loaded after the page renders, so it costs
 * nothing on first paint. The animation pauses offscreen and in hidden tabs, draws a single
 * frame with reduced motion, and uses a coarser grid on small screens.
 */
export function HeroLandscape() {
  const ref = useRef<HTMLCanvasElement>(null);
  const dark = useIsDark();
  const themeRef = useRef<(dark: boolean) => void>();

  useEffect(() => {
    themeRef.current?.(dark);
  }, [dark]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    let disposed = false;
    let cleanup = () => {};

    import("three").then((THREE) => {
      if (disposed) return;
      const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      const small = canvas.clientWidth < 700;

      let renderer: InstanceType<typeof THREE.WebGLRenderer>;
      try {
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      } catch {
        return; // no WebGL: the hero simply has no background
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, small ? 1.5 : 2));
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(small ? 52 : 44, 1, 0.1, 120);

      // ---- value noise, summed over octaves; ridged for the mountains
      const perm = new Uint8Array(512);
      {
        const p = Array.from({ length: 256 }, (_, i) => i);
        let seed = 7;
        for (let i = 255; i > 0; i--) {
          seed = (seed * 16807) % 2147483647;
          const j = seed % (i + 1);
          [p[i], p[j]] = [p[j], p[i]];
        }
        for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
      }
      const rnd = (x: number, y: number) => perm[(perm[x & 255] + y) & 511] / 255;
      const noise = (x: number, y: number) => {
        const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
        const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
        const a = rnd(xi, yi), b = rnd(xi + 1, yi), c = rnd(xi, yi + 1), d = rnd(xi + 1, yi + 1);
        return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
      };

      // ---- the grid: offset rows make triangles
      const GX = small ? 70 : 120, GZ = small ? 46 : 72;
      const SP = small ? 0.86 : 0.5;
      const N = GX * GZ;
      const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
      const grid: [number, number][] = [];
      for (let z = 0; z < GZ; z++)
        for (let x = 0; x < GX; x++)
          grid.push([(x - (GX - 1) / 2) * SP + (z % 2) * SP * 0.5, 6 - z * SP * 0.87]);
      const lines: number[] = [], tris: number[] = [];
      for (let z = 0; z < GZ - 1; z++)
        for (let x = 0; x < GX - 1; x++) {
          const a = z * GX + x, b = a + 1, c = a + GX, d = c + 1;
          lines.push(a, b, a, c);
          if (z % 2) { lines.push(a, d); tris.push(a, d, b, a, c, d); }
          else { lines.push(b, c); tris.push(a, c, b, b, c, d); }
          if (x === GX - 2) lines.push(b, d);
        }
      const position = new THREE.BufferAttribute(pos, 3);
      const color = new THREE.BufferAttribute(col, 3);
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", position);
      lineGeo.setAttribute("color", color);
      lineGeo.setIndex(lines);
      const fillGeo = new THREE.BufferGeometry();
      fillGeo.setAttribute("position", position);
      fillGeo.setIndex(tris);

      const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true });
      const dotMat = new THREE.PointsMaterial({ size: 1.8, sizeAttenuation: false, vertexColors: true, transparent: true });
      // the ground in the page colour hides the lines behind each hill
      const groundMat = new THREE.MeshBasicMaterial({ polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      const WATER = -0.05;
      const waterMat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false });
      const water = new THREE.Mesh(new THREE.PlaneGeometry(90, 60), waterMat);
      water.rotation.x = -Math.PI / 2;
      water.position.set(0, WATER - 0.02, -15);
      const land = new THREE.Group();
      land.add(new THREE.Mesh(fillGeo, groundMat), water, new THREE.LineSegments(lineGeo, lineMat), new THREE.Points(lineGeo, dotMat));
      land.position.y = -2;
      scene.add(land);

      const C = { low: new THREE.Color(), high: new THREE.Color(), snow: new THREE.Color(), water: new THREE.Color() };
      const applyTheme = (isDark: boolean) => {
        const bg = new THREE.Color(isDark ? "#0d1117" : "#ffffff");
        scene.fog = new THREE.Fog(bg, 24, 46);
        groundMat.color.copy(bg);
        C.low.set(isDark ? "#3f8cff" : "#2a78d6");
        C.high.set(isDark ? "#a371f7" : "#8250df");
        C.snow.set(isDark ? "#e6edf3" : "#94a3b8");
        C.water.set(isDark ? "#2dd4bf" : "#0d9488");
        waterMat.color.copy(C.water);
        waterMat.opacity = isDark ? 0.16 : 0.14;
        lineMat.opacity = isDark ? 0.7 : 0.6;
        dotMat.opacity = isDark ? 0.7 : 0.55;
        draw(lastT);
      };

      const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
      const onMove = (e: PointerEvent) => {
        mouse.x = (e.clientX / window.innerWidth - 0.5) * 2;
        mouse.y = (e.clientY / window.innerHeight - 0.5) * 2;
      };
      window.addEventListener("pointermove", onMove);

      const c = new THREE.Color();
      let lastT = 3;
      function draw(t: number) {
        lastT = t;
        const travel = t * 0.9;
        for (let i = 0; i < N; i++) {
          const [x, z] = grid[i];
          const wz = z - travel;
          let hills = 0, amp = 1, f = 0.09;
          for (let o = 0; o < 4; o++) { hills += amp * noise(x * f + 31, wz * f + 17); amp *= 0.5; f *= 2.1; }
          hills /= 1.875;
          const r = 1 - Math.abs(noise(x * 0.06 + 5, wz * 0.06 + 9) * 2 - 1);
          const ridge = r * r;
          const far = Math.min(1, Math.max(0, (-z + 1) / 12));
          // big mountains on the right on wide screens; centred on phones
          const side = small ? 1 : Math.min(1, Math.max(0, (x - 1) / 10));
          let h = (hills - 0.56) * 2.8 + Math.pow(ridge, 1.4) * 6.5 * far * (0.12 + 0.88 * side);
          let wet = false;
          if (h < WATER) {
            wet = true;
            h = WATER + 0.035 * Math.sin(x * 2.1 + t * 1.6) * Math.sin(wz * 1.7 - t * 1.1);
          }
          pos[i * 3] = x; pos[i * 3 + 1] = h; pos[i * 3 + 2] = z;
          if (wet) c.copy(C.water);
          else {
            c.copy(C.low).lerp(C.high, Math.min(1, Math.max(0, h / 4.5)));
            if (h > 3.6) c.lerp(C.snow, Math.min(1, (h - 3.6) / 1.2));
          }
          col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
        }
        position.needsUpdate = true;
        color.needsUpdate = true;
        mouse.sx += (mouse.x - mouse.sx) * 0.03;
        mouse.sy += (mouse.y - mouse.sy) * 0.03;
        const cx = small ? 0 : 1.0;
        camera.position.set(cx + mouse.sx * 0.8, (small ? 3.4 : 2.8) - mouse.sy * 0.3 + Math.sin(t * 0.25) * 0.08, 8.5);
        camera.lookAt((small ? 0 : 2.0) + mouse.sx * 0.4, 0.2, -12);
        renderer.render(scene, camera);
      }

      const resize = () => {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        draw(lastT);
      };
      const ro = new ResizeObserver(resize);
      ro.observe(canvas);

      // run only while visible
      let raf = 0, visible = true, start = performance.now() - lastT * 1000;
      const loop = (now: number) => {
        draw((now - start) / 1000);
        raf = requestAnimationFrame(loop);
      };
      const play = () => {
        if (reduced || raf || !visible || document.hidden) return;
        start = performance.now() - lastT * 1000;
        raf = requestAnimationFrame(loop);
      };
      const stop = () => { cancelAnimationFrame(raf); raf = 0; };
      const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? play() : stop(); });
      io.observe(canvas);
      const onVis = () => (document.hidden ? stop() : play());
      document.addEventListener("visibilitychange", onVis);

      themeRef.current = applyTheme;
      applyTheme(document.documentElement.classList.contains("dark"));
      resize();
      play();

      cleanup = () => {
        stop();
        io.disconnect();
        ro.disconnect();
        document.removeEventListener("visibilitychange", onVis);
        window.removeEventListener("pointermove", onMove);
        lineGeo.dispose(); fillGeo.dispose(); lineMat.dispose(); dotMat.dispose(); groundMat.dispose(); waterMat.dispose();
        renderer.dispose();
      };
    });

    return () => {
      disposed = true;
      cleanup();
    };
  }, []);

  return (
    <canvas
      ref={ref}
      aria-hidden
      className="hero-landscape pointer-events-none absolute inset-x-0 top-0 h-[560px] w-full sm:h-[640px]"
    />
  );
}
