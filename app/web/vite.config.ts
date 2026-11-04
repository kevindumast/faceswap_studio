import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        // Bibliothèques à part : elles changent rarement, le navigateur les garde en cache d'une version à l'autre.
        manualChunks: {
          react: ["react", "react-dom", "react-dom/client"],
          motion: ["motion/react"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
